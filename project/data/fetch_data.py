from gdown import download
import os


def fetch_data():
    cwd = os.path.dirname(os.path.abspath(__file__))

    if not os.path.exists(os.path.join(cwd, 'files')):
        print('Files directory doesn\'t exist, creating...')
        os.makedirs(os.path.join(cwd, 'files'))

    if not os.path.exists(os.path.join(cwd, 'files/ChCh-Miner_durgbank-chem-chem.tsv.gz')):
        download('https://drive.google.com/uc?id=1Ot-ICpiJRlisFvM9Fi6TM3Q6kAZaLS0y', os.path.join(cwd, 'files/ChCh-Miner_durgbank-chem-chem.tsv.gz'))
    if not os.path.exists(os.path.join(cwd, 'files/drugbank_all_full_database.xml.zip')):
        download('https://drive.google.com/uc?id=1LSdAthCa69kWRIKoI5UmclLgf4OsSNAm', os.path.join(cwd, 'files/drugbank_all_full_database.xml.zip'))
    if not os.path.exists(os.path.join(cwd, 'files/drugbank.xsd')):
        download('https://drive.google.com/uc?id=15_hqow9NT_M49OX7cXrG5P6vCgfbKyhP', os.path.join(cwd, 'files/drugbank.xsd'))
