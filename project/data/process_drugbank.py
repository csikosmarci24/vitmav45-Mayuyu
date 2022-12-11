import zipfile
import xml.etree.ElementTree as ET
import collections
import pandas as pd
import os


def collapse_list_values(row):
    for key, value in row.items():
        if isinstance(value, list):
            row[key] = '|'.join(value)
    return row


def parse_drugbank_xml():
    """Unzips the drugbank xml file, and loads its content into a Pandas dataframe. Returns the dataframe object."""
    cwd = os.path.dirname(os.path.abspath(__file__))

    with zipfile.ZipFile(os.path.join(cwd, 'files/drugbank_all_full_database.xml.zip'), 'r').open('full database.xml') as xml_file:
        tree = ET.parse(xml_file)
    root = tree.getroot()

    ns = '{http://www.drugbank.ca}'
    inchikey_template = "{ns}calculated-properties/{ns}property[{ns}kind='InChIKey']/{ns}value"
    inchi_template = "{ns}calculated-properties/{ns}property[{ns}kind='InChI']/{ns}value"
    logP_template = "{ns}calculated-properties/{ns}property[{ns}kind='logP']/{ns}value"
    logS_template = "{ns}calculated-properties/{ns}property[{ns}kind='logS']/{ns}value"
    psa_template = "{ns}calculated-properties/{ns}property[{ns}kind='Polar Surface Area (PSA)']/{ns}value"
    refractivity_template = "{ns}calculated-properties/{ns}property[{ns}kind='Refractivity']/{ns}value"
    polarizability_template = "{ns}calculated-properties/{ns}property[{ns}kind='Polarizability']/{ns}value"
    pka_acidic_template = "{ns}calculated-properties/{ns}property[{ns}kind='pKa (strongest acidic)']/{ns}value"
    pka_basic_template = "{ns}calculated-properties/{ns}property[{ns}kind='pKa (strongest basic)']/{ns}value"
    nor_template = "{ns}calculated-properties/{ns}property[{ns}kind='Number of Rings']/{ns}value"

    rows = list()
    for i, drug in enumerate(root):
        row = collections.OrderedDict()
        assert drug.tag == ns + 'drug'
        row['type'] = drug.get('type')
        row['drugbank_id'] = drug.findtext(ns + "drugbank-id[@primary='true']")
        row['name'] = drug.findtext(ns + "name")
        row['description'] = drug.findtext(ns + "description")
        row['groups'] = [group.text for group in
                         drug.findall("{ns}groups/{ns}group".format(ns=ns))]
        row['atc_codes'] = [code.get('code') for code in
                            drug.findall("{ns}atc-codes/{ns}atc-code".format(ns=ns))]
        row['categories'] = [x.findtext(ns + 'category') for x in
                             drug.findall("{ns}categories/{ns}category".format(ns=ns))]
        row['inchi'] = drug.findtext(inchi_template.format(ns=ns))
        row['logP'] = drug.findtext(logP_template.format(ns=ns))
        row['logS'] = drug.findtext(logS_template.format(ns=ns))
        row['psa'] = drug.findtext(psa_template.format(ns=ns))
        row['refractivity'] = drug.findtext(refractivity_template.format(ns=ns))
        row['polarizability'] = drug.findtext(polarizability_template.format(ns=ns))
        row['pKa_acidic'] = drug.findtext(pka_acidic_template.format(ns=ns))
        row['pKa_basic'] = drug.findtext(pka_basic_template.format(ns=ns))
        row['num_rings'] = drug.findtext(nor_template.format(ns=ns))

        # Add drug aliases
        aliases = {
            elem.text for elem in
            drug.findall("{ns}international-brands/{ns}international-brand".format(ns=ns)) +
            drug.findall("{ns}synonyms/{ns}synonym[@language='English']".format(ns=ns)) +
            drug.findall("{ns}international-brands/{ns}international-brand".format(ns=ns)) +
            drug.findall("{ns}products/{ns}product/{ns}name".format(ns=ns))

        }
        aliases.add(row['name'])
        row['aliases'] = sorted(aliases)

        rows.append(row)

    rows = list(map(collapse_list_values, rows))
    columns = ['drugbank_id', 'name', 'type', 'groups', 'atc_codes', 'categories', 'logP', 'logS', 'psa',
               'refractivity', 'polarizability', 'pKa_acidic', 'pKa_basic', 'num_rings']
    drugbank_df = pd.DataFrame.from_dict(rows)[columns]

    # extracting information from the ATC code, then dropping the atc_codes column
    drugbank_df['ATC1'] = drugbank_df['atc_codes'].astype(str).str[0]
    drugbank_df['ATC2'] = drugbank_df['atc_codes'].astype(str).str[1:3]
    drugbank_df['ATC3'] = drugbank_df['atc_codes'].astype(str).str[3]
    drugbank_df['ATC4'] = drugbank_df['atc_codes'].astype(str).str[4]
    drugbank_df['ATC5'] = drugbank_df['atc_codes'].astype(str).str[5:7]
    drugbank_df = drugbank_df.drop('atc_codes', axis=1)

    # converting the atc codes, types and groups to categorical
    drugbank_df['ATC1'] = drugbank_df['ATC1'].astype('category')
    drugbank_df['ATC2'] = drugbank_df['ATC2'].astype('category')
    drugbank_df['ATC3'] = drugbank_df['ATC3'].astype('category')
    drugbank_df['ATC4'] = drugbank_df['ATC4'].astype('category')
    drugbank_df['ATC5'] = drugbank_df['ATC5'].astype('category')
    drugbank_df['type'] = drugbank_df['type'].astype('category')
    drugbank_df['groups'] = drugbank_df['groups'].astype('category')

    drugbank_df['logP'] = pd.to_numeric(drugbank_df['logP'])
    drugbank_df['logS'] = pd.to_numeric(drugbank_df['logS'])
    drugbank_df['psa'] = pd.to_numeric(drugbank_df['psa'])
    drugbank_df['refractivity'] = pd.to_numeric(drugbank_df['refractivity'])
    drugbank_df['polarizability'] = pd.to_numeric(drugbank_df['polarizability'])
    drugbank_df['pKa_acidic'] = pd.to_numeric(drugbank_df['pKa_acidic'])
    drugbank_df['pKa_basic'] = pd.to_numeric(drugbank_df['pKa_basic'])
    drugbank_df['num_rings'] = pd.to_numeric(drugbank_df['num_rings'])

    # converting the categorical columns to integer columns
    cat_cols = drugbank_df.select_dtypes(['category']).columns
    drugbank_df[cat_cols] = drugbank_df[cat_cols].apply(lambda x: x.cat.codes)
    drugbank_df.set_index('drugbank_id', inplace=True)

    return drugbank_df


if __name__ == "__main__":
    parse_drugbank_xml()
