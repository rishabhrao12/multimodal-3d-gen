import glob
import random
import trimesh
import pyvista as pv
from IPython.display import display
import os
import numpy as np
import pandas as pd
import math

def apply_template(row, template):
    """Applies template to create semantic descriptor used in .apply

    Args:
        row (_type_): dataset row
        template (_type_): template for descriptor

    Returns:
        _type_: string
    """
    tags = row['tags']
    if pd.isna(tags) or tags == "*":
        tags = "<no_tags>"
    elif isinstance(tags, list):
        tags = ", ".join(tags)
    
    synset_words = row['synset words']
    if isinstance(synset_words, list):
        synset_words = [item for item in synset_words if not pd.isna(item)]
        if not synset_words:
            synset_words = "<blank>"
        else:
            synset_words = ", ".join(synset_words)
    
    synset_gloss = row['synset gloss']
    if isinstance(synset_gloss, list):
        synset_gloss = [item for item in synset_gloss if not pd.isna(item)]
        if not synset_gloss:
            synset_gloss = "<blank>"
        else:
            synset_gloss = ", ".join(synset_gloss)
        
    return template.replace("[name]", row['name']).replace("[tags]", tags).replace("[synset words]", synset_words).replace("[synset gloss]", synset_gloss)
    

def create_semantic_descriptor(df):
    """Applies template to create semantic descriptor using tags, name, synset gloss and words

    Args:
        row (_type_): _description_
        template (_type_): _description_

    Returns:
        _type_: pd.DataFrame
    """

    template1 = """A [name] which is commonly known as [tags]. It is associated with the following characteristics: [synset words].
    A general description of this item is: [synset gloss]."""

    template2 = """The [name] is a [synset words] often used for [tags]. It can be described as: [synset gloss]."""

    template3 = """[name] is a [synset words] designed for [tags]. It serves the purpose of [synset gloss]."""

    df['template1_desc'] = df.apply(lambda row: apply_template(row, template1), axis=1)
    df['template2_desc'] = df.apply(lambda row: apply_template(row, template2), axis=1)
    df['template3_desc'] = df.apply(lambda row: apply_template(row, template3), axis=1)
    return df

def create_subset_by_random_sampling(num_cat_sample = 20, meshes_per_cat = 10):
    """Creates a subset of the ShapeNetSem dataset with paired textual modality (semantic descriptor)

    Args:
        num_cat_sample (int, optional): Number of categories for subset. Defaults to 20.
        meshes_per_cat (int, optional): Number of meshes per category. Defaults to 10.

    Returns:
        _type_: Dataset with (num_cat_sample*meshes_per_cat) rows
    """

    metadata_df = pd.read_csv('../Data/ShapeNetSem/Files/metadata.csv')
    categories_df = pd.read_csv('../Data/ShapeNetSem/Files/categories.synset.csv')

    remaining_categories = categories_df['category'].to_list()
    sampled_categories = []
    sampled_metadata_df, metadata_with_category_df, final_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    while len(sampled_categories) < num_cat_sample:
        sampled_category = random.choice(remaining_categories) # select a category at random
        metadata_entries = metadata_df[metadata_df['category'].str.contains(sampled_category, case=False, na=False)] # for cases where there might be more than one category
        num_samples = len(metadata_entries)

        if num_samples >= meshes_per_cat: # make sure there are minimum meshes and only sample in that case
            sample_size = min(meshes_per_cat, len(metadata_entries))
            sampled_metadata_row = metadata_entries.sample(n=sample_size, random_state=42)
            sampled_metadata_df = pd.concat([sampled_metadata_df, sampled_metadata_row])
            sampled_categories.append(sampled_category)

        remaining_categories.remove(sampled_category)

    sampled_metadata_df.reset_index(inplace=True)

    # Joining category info with metadata info
    for _, row in sampled_metadata_df.iterrows():
        # Split the categories by commas (for cases like 'Laptop, PC')
        categories_list = row['category'].split(',')
        synset_words_cat, synset_gloss_cat = [], []
        category_row = pd.DataFrame()
        flag = False
        for category in categories_list:
            category_row = pd.DataFrame()
            if '_' not in category: # No attribute categories
                category = category.strip()
                category_row = categories_df[categories_df['category'] == category] # Find category info
                synset_words = category_row['synset words'].to_list()
                synset_gloss = category_row['synset gloss'].to_list()

                # To check that it has not already been added to the list, gets rid of duplicate category info problem (chair was getting duplicated as same info for multiple categories eg Chair and OfficeChair)
                if not set(synset_words).issubset(set(synset_words_cat)):
                    synset_words_cat.extend(synset_words) # Add info to lists
                if not set(synset_gloss).issubset(set(synset_gloss_cat)):
                    synset_gloss_cat.extend(synset_gloss) # Add info to lists
                flag = True

        if flag:
            # Create a new row with the current metadata and synset info
            expanded_row = row.copy()  # Copy the current row
            expanded_row['synset words'] = synset_words_cat # Use extended list of all sub categories 
            expanded_row['synset gloss'] = synset_gloss_cat # Use extended list of all sub categories
            
            # Append the expanded row to the new DataFrame
            metadata_with_category_df = pd.concat([metadata_with_category_df, expanded_row.to_frame().T], ignore_index=True)
        else:
            print(category, " not available")

    if not metadata_with_category_df.empty:
        final_df = metadata_with_category_df[['fullId', 'category', 'name', 'tags', 'synset words', 'synset gloss']]
        final_df.reset_index()
    
    final_df['fullId'] = final_df['fullId'].apply(lambda x: x.replace('wss.', ''))
    final_df_with_semantic_descriptor = create_semantic_descriptor(final_df)
    return final_df_with_semantic_descriptor

def angle_step_snapshots(mesh_id, mesh_dir, img_dir, num_views=8):
    """Creates snapshots by rotating camera around mesh

    Args:
        mesh_id (_type_): fullID of mesh in dataset (without .wss)
        mesh_dir (_type_): directory of all .obj meshes
        img_dir (_type_): save directory for snapshots
        num_views (int, optional): _description_. Defaults to 8.
    """
    mesh_path = os.path.join(mesh_dir, f"{mesh_id}.obj")
    mesh = pv.read(mesh_path)  # your mesh path
    mesh.rotate_y(90, inplace=True)
    # Center/scale the mesh as before
    x_mid = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
    y_mid = (mesh.bounds[2] + mesh.bounds[3]) / 2.0
    z_mid = (mesh.bounds[4] + mesh.bounds[5]) / 2.0
    mesh.translate([-x_mid, -y_mid, -z_mid], inplace=True)
    max_dim = max(mesh.bounds[1] - mesh.bounds[0],
                mesh.bounds[3] - mesh.bounds[2],
                mesh.bounds[5] - mesh.bounds[4])
    if max_dim > 0:
        scale_factor = 2.0 / max_dim
        mesh.scale(scale_factor, inplace=True)

    plotter = pv.Plotter(notebook=True)
    plotter.add_mesh(mesh, color="lightgray")
    plotter.set_background("white")

    # Position the camera initially
    plotter.camera_position = [
        (0, 0, 5),  # camera location
        (0, 0, 0),  # focal point
        (1, 0, 0)   # 'up' direction
    ]
    #plotter.show(auto_close=False)
    
    mesh_snapshot_dir = os.path.join(img_dir, mesh_id)
    os.makedirs(mesh_snapshot_dir, exist_ok=True)
    
    # Show the first view, saving a screenshot
    #plotter.show(screenshot=f"view0.png", auto_close=False)
    plotter.screenshot(f"{mesh_snapshot_dir}/view0.png")

    angle_step = 360.0 / num_views
    for i in range(1, num_views):
        # Use the VTK method with a capital 'A'
        plotter.camera.Azimuth(angle_step)
        plotter.render()
        # Update the scene + take a screenshot
        #plotter.show(screenshot=f"view{i}.png", auto_close=False)
        plotter.screenshot(f"{mesh_snapshot_dir}/view{i}.png")

def random_camera_snapshots(mesh_id, mesh_dir, img_dir, num_views=8):
    """Creates snapshots by with random camera positions

    Args:
        mesh_id (_type_): fullID of mesh in dataset (without .wss)
        mesh_dir (_type_): directory of all .obj meshes
        img_dir (_type_): save directory for snapshots
        num_views (int, optional): _description_. Defaults to 8.
    """
    mesh_path = os.path.join(mesh_dir, f"{mesh_id}.obj")
    mesh = pv.read(mesh_path)  # your mesh path
    
    # Center/scale the mesh as before
    x_mid = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
    y_mid = (mesh.bounds[2] + mesh.bounds[3]) / 2.0
    z_mid = (mesh.bounds[4] + mesh.bounds[5]) / 2.0
    mesh.translate([-x_mid, -y_mid, -z_mid], inplace=True)
    max_dim = max(mesh.bounds[1] - mesh.bounds[0],
                mesh.bounds[3] - mesh.bounds[2],
                mesh.bounds[5] - mesh.bounds[4])
    if max_dim > 0:
        scale_factor = 2.0 / max_dim
        mesh.scale(scale_factor, inplace=True)

    plotter = pv.Plotter(notebook=True)
    plotter.add_mesh(mesh, color="lightgray")
    plotter.set_background("white")

    mesh_snapshot_dir = os.path.join(img_dir, mesh_id)
    os.makedirs(mesh_snapshot_dir, exist_ok=True)
    
    # Show the first view, saving a screenshot

    for i in range(0, num_views):
        # Use the VTK method with a capital 'A'
        camera_loc = random_point_on_sphere()
        plotter.camera_position = [
            camera_loc,  # camera location
            (0, 0, 0),  # focal point
            (0, 1, 0)   # 'up' direction
        ]
    
        plotter.render()
        # Update the scene + take a screenshot
        #plotter.show(screenshot=f"{mesh_snapshot_dir}/view{i}.png", auto_close=False)
        plotter.screenshot(f"{mesh_snapshot_dir}/view{i}.png")

def random_point_on_sphere(radius=5.0):
    """Return a (x, y, z) coordinate uniformly sampled on a sphere of given radius.

    Args:
        radius (float, optional): _description_. Defaults to 5.0.

    Returns:
        _type_: _description_
    """
    phi = random.uniform(0, 2*math.pi)        # azimuth in [0, 2π)
    costheta = random.uniform(-1, 1)          # cos(theta) in [-1, 1]
    theta = math.acos(costheta)               # polar angle in [0, π]

    x = radius * math.sin(theta) * math.cos(phi)
    y = radius * math.sin(theta) * math.sin(phi)
    z = radius * costheta
    return (x, y, z)

def create_snapshots_for_dataset(dataset_path, mesh_dir, img_dir, num_views):
    """Creates snapshots with multiple views for meshes, this is paired image modality for model

    Args:
        dataset_path (_type_): dataset that has mesh ids for which to create snapshots
        mesh_dir (_type_): read directory for 3D meshes (.obj files)
        img_dir (_type_): save directory for image snapshots
    """
    dataset = pd.read_csv(dataset_path)
    no_errors = 0
    errored_meshes = []
    for index, row in dataset.iterrows():
        try:
            mesh_id = row['fullId']
            angle_step_snapshots(mesh_id, mesh_dir, img_dir, num_views)
        except Exception as e:
            no_errors += 1
            errored_meshes.append(mesh_id)
            print(e)
            continue

    print(f'Snapshots succesfull for {dataset.shape[0] - no_errors} meshes and errors for {no_errors} meshes')
    print(f'Meshes that errored out are: {errored_meshes}')