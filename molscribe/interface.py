import argparse
import time
from typing import List

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from .dataset import get_transforms
from .model import Encoder, Decoder
from .chemistry import convert_graph_to_smiles
from .tokenizer import get_tokenizer


BOND_TYPES = ["", "single", "double", "triple", "aromatic", "solid wedge", "dashed wedge"]


def safe_load(module, module_states):
    def remove_prefix(state_dict):
        return {k.replace('module.', ''): v for k, v in state_dict.items()}
    missing_keys, unexpected_keys = module.load_state_dict(remove_prefix(module_states), strict=False)
    return


class MolScribe:

    def __init__(self, model_path, device=None):
        """
        MolScribe Interface
        :param model_path: path of the model checkpoint.
        :param device: torch device, defaults to be CPU.
        """
        model_states = torch.load(model_path, map_location=torch.device('cpu'))
        args = self._get_args(model_states['args'])
        if device is None:
            device = torch.device('cpu')
        self.device = device
        self.tokenizer = get_tokenizer(args)
        self.encoder, self.decoder = self._get_model(args, self.tokenizer, self.device, model_states)
        self.transform = get_transforms(args.input_size, augment=False)

    def _get_args(self, args_states=None):
        parser = argparse.ArgumentParser()
        # Model
        parser.add_argument('--encoder', type=str, default='swin_base')
        parser.add_argument('--decoder', type=str, default='transformer')
        parser.add_argument('--trunc_encoder', action='store_true')  # use the hidden states before downsample
        parser.add_argument('--no_pretrained', action='store_true')
        parser.add_argument('--use_checkpoint', action='store_true', default=True)
        parser.add_argument('--dropout', type=float, default=0.5)
        parser.add_argument('--embed_dim', type=int, default=256)
        parser.add_argument('--enc_pos_emb', action='store_true')
        group = parser.add_argument_group("transformer_options")
        group.add_argument("--dec_num_layers", help="No. of layers in transformer decoder", type=int, default=6)
        group.add_argument("--dec_hidden_size", help="Decoder hidden size", type=int, default=256)
        group.add_argument("--dec_attn_heads", help="Decoder no. of attention heads", type=int, default=8)
        group.add_argument("--dec_num_queries", type=int, default=128)
        group.add_argument("--hidden_dropout", help="Hidden dropout", type=float, default=0.1)
        group.add_argument("--attn_dropout", help="Attention dropout", type=float, default=0.1)
        group.add_argument("--max_relative_positions", help="Max relative positions", type=int, default=0)
        parser.add_argument('--continuous_coords', action='store_true')
        parser.add_argument('--compute_confidence', action='store_true')
        # Data
        parser.add_argument('--input_size', type=int, default=384)
        parser.add_argument('--vocab_file', type=str, default=None)
        parser.add_argument('--coord_bins', type=int, default=64)
        parser.add_argument('--sep_xy', action='store_true', default=True)

        args = parser.parse_args([])
        if args_states:
            for key, value in args_states.items():
                args.__dict__[key] = value
        return args

    def _get_model(self, args, tokenizer, device, states):
        encoder = Encoder(args, pretrained=False)
        args.encoder_dim = encoder.n_features
        decoder = Decoder(args, tokenizer)

        safe_load(encoder, states['encoder'])
        safe_load(decoder, states['decoder'])
        # print(f"Model loaded from {load_path}")

        encoder.to(device)
        decoder.to(device)
        encoder.eval()
        decoder.eval()
        return encoder, decoder

    def predict_images(self, input_images: List, return_atoms_bonds=False, return_confidence=False, batch_size=16, skip_molblock=False):
        device = self.device
        predictions = []
        self.decoder.compute_confidence = return_confidence

        # Timing data collection
        timing_data = {
            'num_images': len(input_images),
            'batch_size': batch_size,
            'transform_time': 0,
            'stack_to_device_time': 0,
            'encoder_time': 0,
            'decoder_time': 0,
            'smiles_conversion_time': 0,
            'batches': []
        }

        for idx in range(0, len(input_images), batch_size):
            batch_images = input_images[idx:idx+batch_size]
            batch_timing = {'batch_idx': idx // batch_size, 'batch_size': len(batch_images)}

            # Transform timing
            t0 = time.time()
            images = [self.transform(image=image, keypoints=[])['image'] for image in batch_images]
            transform_elapsed = time.time() - t0
            timing_data['transform_time'] += transform_elapsed
            batch_timing['transform_time'] = transform_elapsed

            # Stack and move to device timing
            t0 = time.time()
            images = torch.stack(images, dim=0).to(device)
            stack_elapsed = time.time() - t0
            timing_data['stack_to_device_time'] += stack_elapsed
            batch_timing['stack_to_device_time'] = stack_elapsed

            with torch.no_grad():
                # Encoder timing
                t0 = time.time()
                features, hiddens = self.encoder(images)
                encoder_elapsed = time.time() - t0
                timing_data['encoder_time'] += encoder_elapsed
                batch_timing['encoder_time'] = encoder_elapsed

                # Decoder timing
                t0 = time.time()
                batch_predictions = self.decoder.decode(features, hiddens)
                decoder_elapsed = time.time() - t0
                timing_data['decoder_time'] += decoder_elapsed
                batch_timing['decoder_time'] = decoder_elapsed

            timing_data['batches'].append(batch_timing)
            predictions += batch_predictions

        # SMILES conversion timing
        t0 = time.time()
        outputs = self.convert_graph_to_output(predictions, input_images, return_confidence, return_atoms_bonds, skip_molblock=skip_molblock)
        timing_data['smiles_conversion_time'] = time.time() - t0

        # Store timing for retrieval
        self._last_timing = timing_data

        return outputs

    def get_last_timing(self):
        """Get timing data from the last predict_images call."""
        return getattr(self, '_last_timing', None)


    def convert_graph_to_output(self, predictions, input_images, return_confidence=False, return_atoms_bonds=False, skip_molblock=False):
        t0 = time.time()
        node_coords = [pred['chartok_coords']['coords'] for pred in predictions]
        node_symbols = [pred['chartok_coords']['symbols'] for pred in predictions]
        edges = [pred['edges'] for pred in predictions]
        # node_symbols = [r_groups[symbol] if symbol in r_groups else symbol for symbol in node_symbols]
        import os
        num_workers = int(os.environ.get('MOLSCRIBE_NUM_WORKERS', '1'))
        smiles_list, molblock_list, r_success = convert_graph_to_smiles(
            node_coords, node_symbols, edges, images=input_images, num_workers=num_workers, skip_molblock=skip_molblock)

        outputs = []
        for smiles, molblock, pred in zip(smiles_list, molblock_list, predictions):
            pred_dict = {"smiles": smiles, "molfile": molblock}
            if return_confidence:
                pred_dict["confidence"] = pred["overall_score"]
            if return_atoms_bonds:
                coords = pred['chartok_coords']['coords']
                symbols = pred['chartok_coords']['symbols']
                # get atoms info
                atom_list = []
                for i, (symbol, coord) in enumerate(zip(symbols, coords)):
                    atom_dict = {"atom_symbol": symbol, "x": coord[0], "y": coord[1]}
                    if return_confidence:
                        atom_dict["confidence"] = pred['chartok_coords']['atom_scores'][i]
                    atom_list.append(atom_dict)
                pred_dict["atoms"] = atom_list
                # get bonds info
                bond_list = []
                num_atoms = len(symbols)
                for i in range(num_atoms-1):
                    for j in range(i+1, num_atoms):
                        bond_type_int = pred['edges'][i][j]
                        if bond_type_int != 0:
                            bond_type_str = BOND_TYPES[bond_type_int]
                            bond_dict = {"bond_type": bond_type_str, "endpoint_atoms": (i, j)}
                            if return_confidence:
                                bond_dict["confidence"] = pred["edge_scores"][i][j]
                            bond_list.append(bond_dict)
                pred_dict["bonds"] = bond_list
            outputs.append(pred_dict)

        # Store timing for retrieval
        self._last_convert_timing = {
            'smiles_conversion_time': time.time() - t0,
            'num_graphs': len(predictions)
        }

        return outputs

    def get_last_convert_timing(self):
        """Get timing data from the last convert_graph_to_output call."""
        return getattr(self, '_last_convert_timing', None)

    def predict_image(self, image, return_atoms_bonds=False, return_confidence=False, skip_molblock=False):
        return self.predict_images([
            image], return_atoms_bonds=return_atoms_bonds, return_confidence=return_confidence, skip_molblock=skip_molblock)[0]

    def predict_image_files(self, image_files: List, return_atoms_bonds=False, return_confidence=False, skip_molblock=False):
        input_images = []
        for path in image_files:
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            input_images.append(image)
        return self.predict_images(
            input_images, return_atoms_bonds=return_atoms_bonds, return_confidence=return_confidence, skip_molblock=skip_molblock)

    def predict_image_file(self, image_file: str, return_atoms_bonds=False, return_confidence=False, skip_molblock=False):
        return self.predict_image_files(
            [image_file], return_atoms_bonds=return_atoms_bonds, return_confidence=return_confidence, skip_molblock=skip_molblock)[0]

    def draw_prediction(self, prediction, image, notebook=False):
        if "atoms" not in prediction or "bonds" not in prediction:
            raise ValueError("atoms and bonds information are not provided.")
        h, w, _ = image.shape
        h, w = np.array([h, w]) * 400 / max(h, w)
        image = cv2.resize(image, (int(w), int(h)))
        fig, ax = plt.subplots(1, 1)
        ax.axis('off')
        ax.set_xlim(-0.05 * w, w * 1.05)
        ax.set_ylim(1.05 * h, -0.05 * h)
        plt.imshow(image, alpha=0.)
        x = [a['x'] * w for a in prediction['atoms']]
        y = [a['y'] * h for a in prediction['atoms']]
        markersize = min(w, h) / 3
        plt.scatter(x, y, marker='o', s=markersize, color='lightskyblue', zorder=10)
        for i, atom in enumerate(prediction['atoms']):
            symbol = atom['atom_symbol'].lstrip('[').rstrip(']')
            plt.annotate(symbol, xy=(x[i], y[i]), ha='center', va='center', color='black', zorder=100)
        for bond in prediction['bonds']:
            u, v = bond['endpoint_atoms']
            x1, y1, x2, y2 = x[u], y[u], x[v], y[v]
            bond_type = bond['bond_type']
            if bond_type == 'single':
                color = 'tab:green'
                ax.plot([x1, x2], [y1, y2], color, linewidth=4)
            elif bond_type == 'aromatic':
                color = 'tab:purple'
                ax.plot([x1, x2], [y1, y2], color, linewidth=4)
            elif bond_type == 'double':
                color = 'tab:green'
                ax.plot([x1, x2], [y1, y2], color=color, linewidth=7)
                ax.plot([x1, x2], [y1, y2], color='w', linewidth=1.5, zorder=2.1)
            elif bond_type == 'triple':
                color = 'tab:green'
                x1s, x2s = 0.8 * x1 + 0.2 * x2, 0.2 * x1 + 0.8 * x2
                y1s, y2s = 0.8 * y1 + 0.2 * y2, 0.2 * y1 + 0.8 * y2
                ax.plot([x1s, x2s], [y1s, y2s], color=color, linewidth=9)
                ax.plot([x1, x2], [y1, y2], color='w', linewidth=5, zorder=2.05)
                ax.plot([x1, x2], [y1, y2], color=color, linewidth=2, zorder=2.1)
            else:
                length = 10
                width = 10
                color = 'tab:green'
                if bond_type == 'solid wedge':
                    ax.annotate('', xy=(x1, y1), xytext=(x2, y2),
                                arrowprops=dict(color=color, width=3, headwidth=width, headlength=length), zorder=2)
                else:
                    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                                arrowprops=dict(color=color, width=3, headwidth=width, headlength=length), zorder=2)
        fig.tight_layout()
        if not notebook:
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            buf = canvas.buffer_rgba()
            result_image = np.asarray(buf)
            plt.close(fig)
            return result_image
