import requests

def download(file_with_name, fromURL):
    response = requests.get(fromURL)
    open(file_with_name, "wb").write(response.content)

if __name__ == '__main__':
    download('files/ChCh-Miner_durgbank-chem-chem.tsv.gz', 'https://drive.google.com/uc?id=1Ot-ICpiJRlisFvM9Fi6TM3Q6kAZaLS0y')
    download('files/drugbank_all_full_database.xml.zip', 'https://drive.google.com/uc?id=1LSdAthCa69kWRIKoI5UmclLgf4OsSNAm')
    download('files/drugbank.xsd', 'https://drive.google.com/uc?id=15_hqow9NT_M49OX7cXrG5P6vCgfbKyhP')
