import copy
import traceback
import os
import numpy as np
import multiprocessing

import rdkit
import rdkit.Chem as Chem

rdkit.RDLogger.DisableLog('rdApp.*')

# Default num_workers for multiprocessing pools
# Set MOLSCRIBE_NUM_WORKERS env var to override (default=1 to avoid memory explosion from fork)
_DEFAULT_NUM_WORKERS = int(os.environ.get('MOLSCRIBE_NUM_WORKERS', '1'))

from SmilesPE.pretokenizer import atomwise_tokenizer

from .constants import RGROUP_SYMBOLS, ABBREVIATIONS, VALENCES, FORMULA_REGEX


def is_valid_mol(s, format_='atomtok'):
    if format_ == 'atomtok':
        mol = Chem.MolFromSmiles(s)
    elif format_ == 'inchi':
        if not s.startswith('InChI=1S'):
            s = f"InChI=1S/{s}"
        mol = Chem.MolFromInchi(s)
    else:
        raise NotImplemented
    return mol is not None


def _convert_smiles_to_inchi(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        inchi = Chem.MolToInchi(mol)
    except:
        inchi = None
    return inchi


def convert_smiles_to_inchi(smiles_list, num_workers=_DEFAULT_NUM_WORKERS):
    if num_workers <= 1:
        inchi_list = [_convert_smiles_to_inchi(s) for s in smiles_list]
    else:
        with multiprocessing.Pool(num_workers) as p:
            inchi_list = p.map(_convert_smiles_to_inchi, smiles_list, chunksize=128)
    n_success = sum([x is not None for x in inchi_list])
    r_success = n_success / len(inchi_list)
    inchi_list = [x if x else 'InChI=1S/H2O/h1H2' for x in inchi_list]
    return inchi_list, r_success


def merge_inchi(inchi1, inchi2):
    replaced = 0
    inchi1 = copy.deepcopy(inchi1)
    for i in range(len(inchi1)):
        if inchi1[i] == 'InChI=1S/H2O/h1H2':
            inchi1[i] = inchi2[i]
            replaced += 1
    return inchi1, replaced


def _get_num_atoms(smiles):
    try:
        return Chem.MolFromSmiles(smiles).GetNumAtoms()
    except:
        return 0


def get_num_atoms(smiles, num_workers=_DEFAULT_NUM_WORKERS):
    if type(smiles) is str:
        return _get_num_atoms(smiles)
    if num_workers <= 1:
        num_atoms = [_get_num_atoms(s) for s in smiles]
    else:
        with multiprocessing.Pool(num_workers) as p:
            num_atoms = p.map(_get_num_atoms, smiles)
    return num_atoms


def normalize_nodes(nodes, flip_y=True):
    x, y = nodes[:, 0], nodes[:, 1]
    minx, maxx = min(x), max(x)
    miny, maxy = min(y), max(y)
    x = (x - minx) / max(maxx - minx, 1e-6)
    if flip_y:
        y = (maxy - y) / max(maxy - miny, 1e-6)
    else:
        y = (y - miny) / max(maxy - miny, 1e-6)
    return np.stack([x, y], axis=1)


def _verify_chirality(mol, coords, symbols, edges, debug=False):
    import time as _time
    import sys
    _DIAG = True  # Enable diagnostic logging

    def _log(msg):
        if _DIAG:
            print(f"    [CHIRALITY] {msg}", file=sys.stderr, flush=True)

    try:
        n = mol.GetNumAtoms()
        _log(f"Start: {n} atoms")

        # Make a temp mol to find chiral centers
        mol_tmp = mol.GetMol()
        _log("SanitizeMol #1 (temp)...")
        _t = _time.time()
        Chem.SanitizeMol(mol_tmp)
        _log(f"SanitizeMol #1 done in {_time.time()-_t:.2f}s")

        _log("FindMolChiralCenters...")
        _t = _time.time()
        chiral_centers = Chem.FindMolChiralCenters(
            mol_tmp, includeUnassigned=True, includeCIP=False, useLegacyImplementation=False)
        chiral_center_ids = [idx for idx, _ in chiral_centers]  # List[Tuple[int, any]] -> List[int]
        _log(f"FindMolChiralCenters done in {_time.time()-_t:.2f}s, found {len(chiral_centers)} centers")

        # correction to clear pre-condition violation (for some corner cases)
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.BondType.SINGLE:
                bond.SetBondDir(Chem.BondDir.NONE)

        # Create conformer from 2D coordinate
        conf = Chem.Conformer(n)
        conf.Set3D(True)
        for i, (x, y) in enumerate(coords):
            conf.SetAtomPosition(i, (x, 1 - y, 0))
        mol.AddConformer(conf)
        _log("SanitizeMol #2...")
        _t = _time.time()
        Chem.SanitizeMol(mol)
        _log(f"SanitizeMol #2 done in {_time.time()-_t:.2f}s")
        _log("AssignStereochemistryFrom3D...")
        _t = _time.time()
        Chem.AssignStereochemistryFrom3D(mol)
        _log(f"AssignStereochemistryFrom3D done in {_time.time()-_t:.2f}s")
        # NOTE: seems that only AssignStereochemistryFrom3D can handle double bond E/Z
        # So we do this first, remove the conformer and add back the 2D conformer for chiral correction

        mol.RemoveAllConformers()
        conf = Chem.Conformer(n)
        conf.Set3D(False)
        for i, (x, y) in enumerate(coords):
            conf.SetAtomPosition(i, (x, 1 - y, 0))
        mol.AddConformer(conf)

        # Magic, inferring chirality from coordinates and BondDir. DO NOT CHANGE.
        _log("SanitizeMol #3...")
        _t = _time.time()
        Chem.SanitizeMol(mol)
        _log(f"SanitizeMol #3 done in {_time.time()-_t:.2f}s")
        _log("AssignChiralTypesFromBondDirs...")
        _t = _time.time()
        Chem.AssignChiralTypesFromBondDirs(mol)
        _log(f"AssignChiralTypesFromBondDirs done in {_time.time()-_t:.2f}s")
        _log(f"AssignStereochemistry... (SMILES so far: {Chem.MolToSmiles(mol, isomericSmiles=False)})")
        _t = _time.time()
        Chem.AssignStereochemistry(mol, force=True)
        _log(f"AssignStereochemistry done in {_time.time()-_t:.2f}s")

        # Second loop to reset any wedge/dash bond to be starting from the chiral center)
        for i in chiral_center_ids:
            for j in range(n):
                if edges[i][j] == 5:
                    # assert edges[j][i] == 6
                    mol.RemoveBond(i, j)
                    mol.AddBond(i, j, Chem.BondType.SINGLE)
                    mol.GetBondBetweenAtoms(i, j).SetBondDir(Chem.BondDir.BEGINWEDGE)
                elif edges[i][j] == 6:
                    # assert edges[j][i] == 5
                    mol.RemoveBond(i, j)
                    mol.AddBond(i, j, Chem.BondType.SINGLE)
                    mol.GetBondBetweenAtoms(i, j).SetBondDir(Chem.BondDir.BEGINDASH)
            Chem.AssignChiralTypesFromBondDirs(mol)
            Chem.AssignStereochemistry(mol, force=True)

        # reset chiral tags for non-carbon atom
        for atom in mol.GetAtoms():
            if atom.GetSymbol() != "C":
                atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
        mol = mol.GetMol()

    except Exception as e:
        if debug:
            raise e
        pass
    return mol


def _parse_tokens(tokens: list):
    """
    Parse tokens of condensed formula into list of pairs `(elt, num)`
    where `num` is the multiplicity of the atom (or nested condensed formula) `elt`
    Used by `_parse_formula`, which does the same thing but takes a formula in string form as input
    """
    elements = []
    i = 0
    j = 0
    while i < len(tokens):
        if tokens[i] == '(':
            while j < len(tokens) and tokens[j] != ')':
                j += 1
            elt = _parse_tokens(tokens[i + 1:j])
        else:
            elt = tokens[i]
        j += 1
        if j < len(tokens) and tokens[j].isnumeric():
            num = int(tokens[j])
            j += 1
        else:
            num = 1
        elements.append((elt, num))
        i = j
    return elements


def _parse_formula(formula: str):
    """
    Parse condensed formula into list of pairs `(elt, num)`
    where `num` is the subscript to the atom (or nested condensed formula) `elt`
    Example: "C2H4O" -> [('C', 2), ('H', 4), ('O', 1)]
    """
    tokens = FORMULA_REGEX.findall(formula)
    # if ''.join(tokens) != formula:
    #     tokens = FORMULA_REGEX_BACKUP.findall(formula)
    return _parse_tokens(tokens)


def _expand_carbon(elements: list):
    """
    Given list of pairs `(elt, num)`, output single list of all atoms in order,
    expanding carbon sequences (CaXb where a > 1 and X is halogen) if necessary
    Example: [('C', 2), ('H', 4), ('O', 1)] -> ['C', 'H', 'H', 'C', 'H', 'H', 'O'])
    """
    expanded = []
    i = 0
    while i < len(elements):
        elt, num = elements[i]
        # expand carbon sequence
        if elt == 'C' and num > 1 and i + 1 < len(elements):
            next_elt, next_num = elements[i + 1]
            quotient, remainder = next_num // num, next_num % num
            for _ in range(num):
                expanded.append('C')
                for _ in range(quotient):
                    expanded.append(next_elt)
            for _ in range(remainder):
                expanded.append(next_elt)
            i += 2
        # recurse if `elt` itself is a list (nested formula)
        elif isinstance(elt, list):
            new_elt = _expand_carbon(elt)
            for _ in range(num):
                expanded.append(new_elt)
            i += 1
        # simplest case: simply append `elt` `num` times
        else:
            for _ in range(num):
                expanded.append(elt)
            i += 1
    return expanded


def _expand_abbreviation(abbrev):
    """
    Expand abbreviation into its SMILES; also converts [Rn] to [n*]
    Used in `_condensed_formula_list_to_smiles` when encountering abbrev. in condensed formula
    """
    if abbrev in ABBREVIATIONS:
        return ABBREVIATIONS[abbrev].smiles
    if abbrev in RGROUP_SYMBOLS or (abbrev[0] == 'R' and abbrev[1:].isdigit()):
        if abbrev[1:].isdigit():
            return f'[{abbrev[1:]}*]'
        return '*'
    return f'[{abbrev}]'


def _get_bond_symb(bond_num):
    """
    Get SMILES symbol for a bond given bond order
    Used in `_condensed_formula_list_to_smiles` while writing the SMILES string
    """
    if bond_num == 0:
        return '.'
    if bond_num == 1:
        return ''
    if bond_num == 2:
        return '='
    if bond_num == 3:
        return '#'
    return ''


def _condensed_formula_list_to_smiles(formula_list, start_bond, end_bond=None, direction=None):
    """
    Converts condensed formula (in the form of a list of symbols) to smiles
    Input:
    `formula_list`: e.g. ['C', 'H', 'H', 'N', ['C', 'H', 'H', 'H'], ['C', 'H', 'H', 'H']] for CH2N(CH3)2
    `start_bond`: # bonds attached to beginning of formula
    `end_bond`: # bonds attached to end of formula (deduce automatically if None)
    `direction` (1, -1, or None): direction in which to process the list (1: left to right; -1: right to left; None: deduce automatically)
    Returns:
    `smiles`: smiles corresponding to input condensed formula
    `bonds_left`: bonds remaining at the end of the formula (for connecting back to main molecule); should equal `end_bond` if specified
    `num_trials`: number of trials
    `success` (bool): whether conversion was successful
    """
    # `direction` not specified: try left to right; if fails, try right to left
    if direction is None:
        num_trials = 1
        for dir_choice in [1, -1]:
            smiles, bonds_left, trials, success = _condensed_formula_list_to_smiles(formula_list, start_bond, end_bond, dir_choice)
            num_trials += trials
            if success:
                return smiles, bonds_left, num_trials, success
        return None, None, num_trials, False
    assert direction == 1 or direction == -1

    def dfs(smiles, bonds_left, cur_idx, add_idx):
        """
        `smiles`: SMILES string so far
        `cur_idx`: index (in list `formula`) of current atom (i.e. atom to which subsequent atoms are being attached)
        `cur_flat_idx`: index of current atom in list of atom tokens of SMILES so far
        `bonds_left`: bonds remaining on current atom for subsequent atoms to be attached to
        `add_idx`: index (in list `formula`) of atom to be attached to current atom
        `add_flat_idx`: index of atom to be added in list of atom tokens of SMILES so far
        Note: "atom" could refer to nested condensed formula (e.g. CH3 in CH2N(CH3)2)
        """
        num_trials = 1
        # end of formula: return result
        if (direction == 1 and add_idx == len(formula_list)) or (direction == -1 and add_idx == -1):
            if end_bond is not None and end_bond != bonds_left:
                return smiles, bonds_left, num_trials, False
            return smiles, bonds_left, num_trials, True

        # no more bonds but there are atoms remaining: conversion failed
        if bonds_left <= 0:
            return smiles, bonds_left, num_trials, False
        to_add = formula_list[add_idx]  # atom to be added to current atom

        if isinstance(to_add, list):  # "atom" added is a list (i.e. nested condensed formula): assume valence of 1
            if bonds_left > 1:
                # "atom" added does not use up remaining bonds of current atom
                # get smiles of "atom" (which is itself a condensed formula)
                add_str, val, trials, success = _condensed_formula_list_to_smiles(to_add, 1, None, direction)
                if val > 0:
                    add_str = _get_bond_symb(val + 1) + add_str
                num_trials += trials
                if not success:
                    return smiles, bonds_left, num_trials, False
                # put smiles of "atom" in parentheses and append to smiles; go to next atom to add to current atom
                result = dfs(smiles + f'({add_str})', bonds_left - 1, cur_idx, add_idx + direction)
            else:
                # "atom" added uses up remaining bonds of current atom
                # get smiles of "atom" and bonds left on it
                add_str, bonds_left, trials, success = _condensed_formula_list_to_smiles(to_add, 1, None, direction)
                num_trials += trials
                if not success:
                    return smiles, bonds_left, num_trials, False
                # append smiles of "atom" (without parentheses) to smiles; it becomes new current atom
                result = dfs(smiles + add_str, bonds_left, add_idx, add_idx + direction)
            smiles, bonds_left, trials, success = result
            num_trials += trials
            return smiles, bonds_left, num_trials, success

        # atom added is a single symbol (as opposed to nested condensed formula)
        for val in VALENCES.get(to_add, [1]):  # try all possible valences of atom added
            add_str = _expand_abbreviation(to_add)  # expand to smiles if symbol is abbreviation
            if bonds_left > val:  # atom added does not use up remaining bonds of current atom; go to next atom to add to current atom
                if cur_idx >= 0:
                    add_str = _get_bond_symb(val) + add_str
                result = dfs(smiles + f'({add_str})', bonds_left - val, cur_idx, add_idx + direction)
            else:  # atom added uses up remaining bonds of current atom; it becomes new current atom
                if cur_idx >= 0:
                    add_str = _get_bond_symb(bonds_left) + add_str
                result = dfs(smiles + add_str, val - bonds_left, add_idx, add_idx + direction)
            trials, success = result[2:]
            num_trials += trials
            if success:
                return result[0], result[1], num_trials, success
            if num_trials > 10000:
                break
        return smiles, bonds_left, num_trials, False

    cur_idx = -1 if direction == 1 else len(formula_list)
    add_idx = 0 if direction == 1 else len(formula_list) - 1
    return dfs('', start_bond, cur_idx, add_idx)


def get_smiles_from_symbol(symbol, mol, atom, bonds):
    """
    Convert symbol (abbrev. or condensed formula) to smiles
    If condensed formula, determine parsing direction and num. bonds on each side using coordinates
    """
    if symbol in ABBREVIATIONS:
        return ABBREVIATIONS[symbol].smiles
    if len(symbol) > 20:
        return None
    # Skip symbols with unreasonably large subscripts - these are OCR garbage
    # Subscripts come AFTER element symbols (e.g., C12, H26) and should be small
    # Element symbols: one uppercase, or uppercase+lowercase (e.g., C, Au, Fe)
    import re
    for match in re.finditer(r'(?:[A-Z][a-z]?|[a-z])(\d+)', symbol):
        # Number after element symbol = subscript, limit to 50
        if int(match.group(1)) > 50:
            return None

    total_bonds = int(sum([bond.GetBondTypeAsDouble() for bond in bonds]))
    formula_list = _expand_carbon(_parse_formula(symbol))
    smiles, bonds_left, num_trails, success = _condensed_formula_list_to_smiles(formula_list, total_bonds, None)
    if success:
        return smiles
    return None


def _replace_functional_group(smiles):
    smiles = smiles.replace('<unk>', 'C')
    for i, r in enumerate(RGROUP_SYMBOLS):
        symbol = f'[{r}]'
        if symbol in smiles:
            if r[0] == 'R' and r[1:].isdigit():
                smiles = smiles.replace(symbol, f'[{int(r[1:])}*]')
            else:
                smiles = smiles.replace(symbol, '*')
    # For unknown tokens (i.e. rdkit cannot parse), replace them with [{isotope}*], where isotope is an identifier.
    tokens = atomwise_tokenizer(smiles)
    new_tokens = []
    mappings = {}  # isotope : symbol
    isotope = 50
    for token in tokens:
        if token[0] == '[':
            if token[1:-1] in ABBREVIATIONS or Chem.AtomFromSmiles(token) is None:
                while f'[{isotope}*]' in smiles or f'[{isotope}*]' in new_tokens:
                    isotope += 1
                placeholder = f'[{isotope}*]'
                mappings[isotope] = token[1:-1]
                new_tokens.append(placeholder)
                continue
        new_tokens.append(token)
    smiles = ''.join(new_tokens)
    return smiles, mappings


def convert_smiles_to_mol(smiles):
    if smiles is None or smiles == '':
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
    except:
        return None
    return mol


BOND_TYPES = {1: Chem.rdchem.BondType.SINGLE, 2: Chem.rdchem.BondType.DOUBLE, 3: Chem.rdchem.BondType.TRIPLE}


def _expand_functional_group(mol, mappings, debug=False):
    import sys as _sys
    def _log3(msg):
        print(f"      [EXPAND] {msg}", file=_sys.stderr, flush=True)

    def _need_expand(mol, mappings):
        return any([len(Chem.GetAtomAlias(atom)) > 0 for atom in mol.GetAtoms()]) or len(mappings) > 0

    if _need_expand(mol, mappings):
        mol_w = Chem.RWMol(mol)
        num_atoms = mol_w.GetNumAtoms()
        _log3(f"Need expansion: {num_atoms} atoms")
        for i, atom in enumerate(mol_w.GetAtoms()):  # reset radical electrons
            atom.SetNumRadicalElectrons(0)

        atoms_to_remove = []
        for i in range(num_atoms):
            atom = mol_w.GetAtomWithIdx(i)
            if atom.GetSymbol() == '*':
                symbol = Chem.GetAtomAlias(atom)
                isotope = atom.GetIsotope()
                if isotope > 0 and isotope in mappings:
                    symbol = mappings[isotope]
                if not (isinstance(symbol, str) and len(symbol) > 0):
                    continue
                # rgroups do not need to be expanded
                if symbol in RGROUP_SYMBOLS:
                    _log3(f"Atom {i}: symbol='{symbol}' is RGROUP, skipping")
                    continue

                bonds = atom.GetBonds()
                _log3(f"Atom {i}: symbol='{symbol}', {len(list(bonds))} bonds, calling get_smiles_from_symbol...")
                bonds = atom.GetBonds()  # re-get iterator
                sub_smiles = get_smiles_from_symbol(symbol, mol_w, atom, bonds)
                _log3(f"Atom {i}: get_smiles_from_symbol returned '{sub_smiles}'")

                # create mol object for abbreviation/condensed formula from its SMILES
                mol_r = convert_smiles_to_mol(sub_smiles)

                if mol_r is None:
                    # atom.SetAtomicNum(6)
                    atom.SetIsotope(0)
                    continue

                # remove bonds connected to abbreviation/condensed formula
                adjacent_indices = [bond.GetOtherAtomIdx(i) for bond in bonds]
                for adjacent_idx in adjacent_indices:
                    mol_w.RemoveBond(i, adjacent_idx)

                adjacent_atoms = [mol_w.GetAtomWithIdx(adjacent_idx) for adjacent_idx in adjacent_indices]
                for adjacent_atom, bond in zip(adjacent_atoms, bonds):
                    adjacent_atom.SetNumRadicalElectrons(int(bond.GetBondTypeAsDouble()))

                # get indices of atoms of main body that connect to substituent
                bonding_atoms_w = adjacent_indices
                # assume indices are concated after combine mol_w and mol_r
                bonding_atoms_r = [mol_w.GetNumAtoms()]
                for atm in mol_r.GetAtoms():
                    if atm.GetNumRadicalElectrons() and atm.GetIdx() > 0:
                        bonding_atoms_r.append(mol_w.GetNumAtoms() + atm.GetIdx())

                # combine main body and substituent into a single molecule object
                combo = Chem.CombineMols(mol_w, mol_r)

                # connect substituent to main body with bonds
                mol_w = Chem.RWMol(combo)
                # if len(bonding_atoms_r) == 1:  # substituent uses one atom to bond to main body
                for atm in bonding_atoms_w:
                    bond_order = mol_w.GetAtomWithIdx(atm).GetNumRadicalElectrons()
                    mol_w.AddBond(atm, bonding_atoms_r[0], order=BOND_TYPES[bond_order])

                # reset radical electrons
                for atm in bonding_atoms_w:
                    mol_w.GetAtomWithIdx(atm).SetNumRadicalElectrons(0)
                for atm in bonding_atoms_r:
                    mol_w.GetAtomWithIdx(atm).SetNumRadicalElectrons(0)
                atoms_to_remove.append(i)

        # Remove atom in the end, otherwise the id will change
        # Reverse the order and remove atoms with larger id first
        atoms_to_remove.sort(reverse=True)
        for i in atoms_to_remove:
            mol_w.RemoveAtom(i)
        smiles = Chem.MolToSmiles(mol_w)
        mol = mol_w.GetMol()
    else:
        smiles = Chem.MolToSmiles(mol)
    return smiles, mol


def _convert_graph_to_smiles(coords, symbols, edges, image=None, skip_molblock=False, debug=False):
    mol = Chem.RWMol()
    n = len(symbols)
    ids = []
    for i in range(n):
        symbol = symbols[i]
        if symbol[0] == '[':
            symbol = symbol[1:-1]
        if symbol in RGROUP_SYMBOLS:
            atom = Chem.Atom("*")
            if symbol[0] == 'R' and symbol[1:].isdigit():
                atom.SetIsotope(int(symbol[1:]))
            Chem.SetAtomAlias(atom, symbol)
        elif symbol in ABBREVIATIONS:
            atom = Chem.Atom("*")
            Chem.SetAtomAlias(atom, symbol)
        else:
            try:  # try to get SMILES of atom
                atom = Chem.AtomFromSmiles(symbols[i])
                atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
            except:  # otherwise, abbreviation or condensed formula
                atom = Chem.Atom("*")
                Chem.SetAtomAlias(atom, symbol)

        if atom.GetSymbol() == '*':
            atom.SetProp('molFileAlias', symbol)

        idx = mol.AddAtom(atom)
        assert idx == i
        ids.append(idx)

    for i in range(n):
        for j in range(i + 1, n):
            if edges[i][j] == 1:
                mol.AddBond(ids[i], ids[j], Chem.BondType.SINGLE)
            elif edges[i][j] == 2:
                mol.AddBond(ids[i], ids[j], Chem.BondType.DOUBLE)
            elif edges[i][j] == 3:
                mol.AddBond(ids[i], ids[j], Chem.BondType.TRIPLE)
            elif edges[i][j] == 4:
                mol.AddBond(ids[i], ids[j], Chem.BondType.AROMATIC)
            elif edges[i][j] == 5:
                mol.AddBond(ids[i], ids[j], Chem.BondType.SINGLE)
                mol.GetBondBetweenAtoms(ids[i], ids[j]).SetBondDir(Chem.BondDir.BEGINWEDGE)
            elif edges[i][j] == 6:
                mol.AddBond(ids[i], ids[j], Chem.BondType.SINGLE)
                mol.GetBondBetweenAtoms(ids[i], ids[j]).SetBondDir(Chem.BondDir.BEGINDASH)

    pred_smiles = '<invalid>'
    import sys as _sys

    def _log2(msg):
        print(f"    [CONVERT] {msg}", file=_sys.stderr, flush=True)

    try:
        # TODO: move to an util function
        if image is not None:
            height, width, _ = image.shape
            ratio = width / height
            coords = [[x * ratio * 10, y * 10] for x, y in coords]
        _log2("_verify_chirality...")
        mol = _verify_chirality(mol, coords, symbols, edges, debug)
        _log2("_verify_chirality done")
        # Skip molblock if not needed (avoids potential hang on malformed molecules)
        # Also skip for very large molecules (>100 atoms) - likely garbage, not worth waiting for timeout
        if skip_molblock or n > 100:
            if n > 100:
                _log2(f"Skipping MolToMolBlock for large molecule ({n} atoms)")
            pred_molblock = ''
        else:
            # molblock is obtained before expanding func groups, otherwise the expanded group won't have coordinates.
            # Use timeout to avoid hanging on malformed molecules
            _log2("MolToMolBlock...")
            import multiprocessing
            MOLBLOCK_TIMEOUT = 5  # seconds
            try:
                pool = multiprocessing.Pool(1)
                async_result = pool.apply_async(Chem.MolToMolBlock, (mol,))
                pred_molblock = async_result.get(timeout=MOLBLOCK_TIMEOUT)
                pool.close()
                pool.join()
            except multiprocessing.TimeoutError:
                _log2(f"MolToMolBlock TIMEOUT after {MOLBLOCK_TIMEOUT}s - skipping")
                pool.terminate()
                pool.join()
                pred_molblock = ''
            except Exception as e:
                _log2(f"MolToMolBlock ERROR: {e}")
                try:
                    pool.terminate()
                    pool.join()
                except:
                    pass
                pred_molblock = ''
            else:
                _log2("MolToMolBlock done")
        _log2("_expand_functional_group...")
        pred_smiles, mol = _expand_functional_group(mol, {}, debug)
        _log2(f"_expand_functional_group done, SMILES={pred_smiles[:50] if pred_smiles else 'None'}")
        success = True
    except Exception as e:
        if debug:
            print(traceback.format_exc())
        pred_molblock = ''
        success = False

    if debug:
        return pred_smiles, pred_molblock, mol, success
    return pred_smiles, pred_molblock, success


def convert_graph_to_smiles(coords, symbols, edges, images=None, num_workers=_DEFAULT_NUM_WORKERS, skip_molblock=False):
    # Handle empty input case
    if len(coords) == 0:
        return [], [], 0.0

    if images is None:
        args = [(c, s, e, None, skip_molblock) for c, s, e in zip(coords, symbols, edges)]
    else:
        args = [(c, s, e, img, skip_molblock) for c, s, e, img in zip(coords, symbols, edges, images)]

    # Skip multiprocessing when num_workers <= 1 to avoid fork() memory copy
    if num_workers <= 1:
        print(f"[DEBUG] convert_graph_to_smiles: sequential mode, {len(args)} molecules", flush=True)
        results = []
        import time as _time

        for i, a in enumerate(args):
            coords_i, symbols_i, edges_i, img_i, skip_mb = a
            num_atoms = len(symbols_i) if symbols_i else 0

            print(f"  [MOL {i+1}/{len(args)}] Processing: {num_atoms} atoms", flush=True)
            _start = _time.time()

            try:
                result = _convert_graph_to_smiles(*a)
                _elapsed = _time.time() - _start
                smiles_out = result[0] if result[0] else "(empty)"
                success = result[2]
                print(f"  [MOL {i+1}/{len(args)}] Done in {_elapsed:.2f}s, success={success}, SMILES={smiles_out[:50] if len(smiles_out) > 50 else smiles_out}", flush=True)
                results.append(result)
            except Exception as e:
                _elapsed = _time.time() - _start
                print(f"  [MOL {i+1}/{len(args)}] ERROR after {_elapsed:.1f}s: {e}", flush=True)
                results.append(('', '', False))
    else:
        with multiprocessing.Pool(num_workers) as p:
            results = p.starmap(_convert_graph_to_smiles, args, chunksize=128)

    smiles_list, molblock_list, success = zip(*results)
    r_success = np.mean(success)
    return smiles_list, molblock_list, r_success


def _postprocess_smiles(smiles, coords=None, symbols=None, edges=None, molblock=False, debug=False):
    if type(smiles) is not str or smiles == '':
        return '', False
    mol = None
    pred_molblock = ''
    try:
        pred_smiles = smiles
        pred_smiles, mappings = _replace_functional_group(pred_smiles)
        if coords is not None and symbols is not None and edges is not None:
            pred_smiles = pred_smiles.replace('@', '').replace('/', '').replace('\\', '')
            mol = Chem.RWMol(Chem.MolFromSmiles(pred_smiles, sanitize=False))
            mol = _verify_chirality(mol, coords, symbols, edges, debug)
        else:
            mol = Chem.MolFromSmiles(pred_smiles, sanitize=False)
        # pred_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        if molblock:
            pred_molblock = Chem.MolToMolBlock(mol)
        pred_smiles, mol = _expand_functional_group(mol, mappings)
        success = True
    except Exception as e:
        if debug:
            print(traceback.format_exc())
        pred_smiles = smiles
        pred_molblock = ''
        success = False
    if debug:
        return pred_smiles, pred_molblock, mol, success
    return pred_smiles, pred_molblock, success


def postprocess_smiles(smiles, coords=None, symbols=None, edges=None, molblock=False, num_workers=_DEFAULT_NUM_WORKERS):
    # Handle empty input case
    if len(smiles) == 0:
        return [], [], 0.0

    if coords is not None and symbols is not None and edges is not None:
        args = list(zip(smiles, coords, symbols, edges))
        if num_workers <= 1:
            results = [_postprocess_smiles(*a) for a in args]
        else:
            with multiprocessing.Pool(num_workers) as p:
                results = p.starmap(_postprocess_smiles, args, chunksize=128)
    else:
        if num_workers <= 1:
            results = [_postprocess_smiles(s) for s in smiles]
        else:
            with multiprocessing.Pool(num_workers) as p:
                results = p.map(_postprocess_smiles, smiles, chunksize=128)

    smiles_list, molblock_list, success = zip(*results)
    r_success = np.mean(success)
    return smiles_list, molblock_list, r_success


def _keep_main_molecule(smiles, debug=False):
    try:
        mol = Chem.MolFromSmiles(smiles)
        frags = Chem.GetMolFrags(mol, asMols=True)
        if len(frags) > 1:
            num_atoms = [m.GetNumAtoms() for m in frags]
            main_mol = frags[np.argmax(num_atoms)]
            smiles = Chem.MolToSmiles(main_mol)
    except Exception as e:
        if debug:
            print(traceback.format_exc())
    return smiles


def keep_main_molecule(smiles, num_workers=_DEFAULT_NUM_WORKERS):
    if num_workers <= 1:
        results = [_keep_main_molecule(s) for s in smiles]
    else:
        with multiprocessing.Pool(num_workers) as p:
            results = p.map(_keep_main_molecule, smiles, chunksize=128)
    return results
