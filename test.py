import h5py

# Ruta al archivo del modelo
model_path = 'pretrained-models/flickr_style.h5'

# Abrir el archivo .h5 y listar su contenido
with h5py.File(model_path, 'r') as f:
    print("Keys in the .h5 file:")
    print(list(f.keys()))

    # Si hay un grupo llamado 'model_weights', lo inspeccionamos más a fondo
    if 'model_weights' in f.keys():
        print("\nKeys in model_weights group:")
        print(list(f['model_weights'].keys()))

    # Inspeccionamos el grupo de configuraciones del modelo
    if 'model_config' in f.keys():
        print("\nModel config:")
        print(f['model_config'][()])

    # Inspeccionamos el grupo de atributos del modelo
    if 'attributes' in f.keys():
        print("\nAttributes:")
        print(list(f['attributes'].keys()))
