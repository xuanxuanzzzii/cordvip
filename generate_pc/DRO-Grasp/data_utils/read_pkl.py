import pickle

def read_and_print_pkl_file(pkl_file_path):
    try:
        with open(pkl_file_path, 'rb') as f:
            data = pickle.load(f)
            print(f"Content of {pkl_file_path}:")
            print(data)
            print('object',data['object_cloud'].shape)
            print('hand',data['hand_cloud'].shape)
            print("-" * 50)
    except Exception as e:
        print(f"Error reading {pkl_file_path}: {e}")

pkl_file_path = './expert_data/flip_cup/recorded_data/demonstration_37/210'
read_and_print_pkl_file(pkl_file_path)
