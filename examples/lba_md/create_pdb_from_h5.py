import pickle
import h5py
import argparse


atomic_numbers_Map = {1:'H', 5:'B', 6:'C', 7:'N', 8:'O',11:'Na',12:'Mg',14:'Si',15:'P',16:'S',17:'Cl',19:'K',20:'Ca',35:'Br',53:'I'}

def get_maps():
    residueMap = pickle.load(open('atoms_residue_map.pickle','rb'))
    typeMap = pickle.load(open('atoms_type_map.pickle','rb'))
    nameMap = pickle.load(open('atoms_name_map_for_pdb.pickle','rb'))
    return residueMap, typeMap, nameMap

def get_entries(struct):
    trajectory_coordinates = f.get(struct+'/'+'trajectory_coordinates')
    atoms_type = f.get(struct+'/'+'atoms_type')    
    atoms_number = f.get(struct+'/'+'atoms_number') 
    atoms_residue = f.get(struct+'/'+'atoms_residue') 
    molecules_begin_atom_index = f.get(struct+'/'+'molecules_begin_atom_index') 
    return trajectory_coordinates,atoms_type,atoms_number,atoms_residue,molecules_begin_atom_index

def get_atom_name(i, atoms_number, residue_atom_index, residue_name, type_string):
    if residue_name == 'MOL':
        atom_name = atomic_numbers_Map[atoms_number[i]]+str(residue_atom_index)
    else:
        try:
            atom_name = nameMap[(residue_name, residue_atom_index-1, type_string)]
        except KeyError:
            print('KeyError', (residue_name, residue_atom_index-1, type_string))
            atom_name = atomic_numbers_Map[atoms_number[i]]+str(residue_atom_index)
    return atom_name

def update_residue_indices(residue_number, i, type_string, atoms_type, atoms_residue, residue_name, residue_atom_index):
    """
    If the atom sequence has O-N icnrease the residueNumber
    """
    if i < len(atoms_type)-1:
        if type_string == 'O' and typeMap[atoms_type[i+1]] == 'N' or residue_Map[atoms_residue[i+1]]=='MOL':
            # GLN has a O N sequence within the AA
            if not ((residue_name == 'GLN' and residue_atom_index==12) or (residue_name == 'ASN' and residue_atom_index==9)):
                residue_number +=1
                residue_atom_index = 0
    return residue_number, residue_atom_index

def insert_TERS(i, molecules_begin_atom_index, residue_number, residue_atom_index, lines):
    """
    We have to insert TERs for the endings of the molecule
    """
    if i+1 in molecules_begin_atom_index:
        lines.append('TER')
        residue_number +=1
        residue_atom_index = 0
    return residue_number, residue_atom_index, lines

def create_pdb_lines(frame, trajectory_coordinates, atoms_type, atoms_number, atoms_residue, molecules_begin_atom_index):
    """
    We go through each atom line and bring the inputs in the pdb format
    
    """
    lines = []
    residue_number = 1
    residue_atom_index = 0
    for i in range(len(atoms_type)):
        residue_atom_index +=1
        type_string = typeMap[atoms_type[i]]
        residue_name = residue_Map[atoms_residue[i]]
        atom_name = get_atom_name(i, atoms_number, residue_atom_index, residue_name, type_string)
        x,y,z = trajectory_coordinates[frame][i][0],trajectory_coordinates[frame][i][1],trajectory_coordinates[frame][i][2]
        line = 'ATOM{0:7d}  {1:<4}{2:<4}{3:>5}    {4:8.3f}{5:8.3f}{6:8.3f}  1.00  0.00           {7:<5}'.format(i+1,atom_name,residue_name,residue_number,x,y,z,atomic_numbers_Map[atoms_number[i]])
        residue_number, residue_atom_index = update_residue_indices(residue_number, i, type_string, atoms_type, atoms_residue, residue_name, residue_atom_index)
        lines.append(line)
        residue_number, residue_atom_index, lines = insert_TERS(i, molecules_begin_atom_index, residue_number, residue_atom_index, lines)
    return lines

def write_pdb(struct, frame, lines):
    with open(struct+'_frame'+str(frame)+'.pdb', 'w') as of:
        for line in lines:
            of.write(line+'\n')



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-struct", "--struct", required=True, help="pdb code of struct to convert e.g. 11gs")
    parser.add_argument("-frame", "--frame", required=False, help="Frame of trajectory to convert", default=0, type=int)
    parser.add_argument("-dataset", "--dataset", required=False, help="Dataset in hdf5 format", default='MD_dataset_mapped.hdf5', type=str)
    args = parser.parse_args()

    f = h5py.File(args.dataset, 'r')
    struct = args.struct
    frame = args.frame

    residue_Map, typeMap, nameMap = get_maps()
    trajectory_coordinates, atoms_type, atoms_number, atoms_residue, molecules_begin_atom_index = get_entries(struct)
    print('Starting conversion of '+struct+' frame '+str(args.frame))
    lines = create_pdb_lines(frame, trajectory_coordinates, atoms_type, atoms_number, atoms_residue, molecules_begin_atom_index)
    write_pdb(struct, frame, lines)







