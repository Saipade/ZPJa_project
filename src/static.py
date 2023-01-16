import os
models_dir = os.path.join(os.path.abspath(__file__).rsplit('/', maxsplit=2)[0], 'models')
laser_path = os.path.join(models_dir, 'laser')
fasttext_path = os.path.join(models_dir, 'fasttext')
spelling_dict_path = os.path.join(models_dir, 'spelling')
