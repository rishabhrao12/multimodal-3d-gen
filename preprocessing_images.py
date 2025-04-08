from create_dataset import *

if __name__ == "__main__":
    dataset_path = 'Data/ShapeNetSem/Datasets/baseline_400.csv'
    mesh_dir = "Data/ShapeNetSem/Files/models-OBJ/models/"
    img_dir = "Data/ShapeNetSem/Images/baseline_400/"
    num_views = 4
    os.makedirs(img_dir, exist_ok=True)
    create_snapshots_for_dataset(dataset_path, mesh_dir, img_dir, num_views)