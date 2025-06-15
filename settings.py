## feature names
#lst_findings=[
#    'Change, basophilic',
#    'Change, acidophilic',
#    'Fibrosis',
#    'Ground glass appearance',
#    'Degeneration, hydropic',
#    'Cellular infiltration, mononuclear cell',
#    'Microgranuloma',
#    'Increased mitosis',
#    'Single cell necrosis',
#    'Swelling',
#    'Hypertrophy',
#    'Inclusion body, intracytoplasmic',
#    'Change, eosinophilic',
#    'Proliferation, Kupffer cell',
#    'Cellular infiltration',
#    'Degeneration, fatty',
#    'Vacuolization, cytoplasmic',
#    'Anisonucleosis',
#    'Degeneration, granular, eosinophilic',
#    'Hematopoiesis, extramedullary',
#    'Necrosis',
#    'Proliferation, bile duct']
lst_findings=[
    'Hepatocellular Degeneration',
    'Hepatocellular Injury, and Death',
    'Billary Change',
    'Hepatocellular Responses',
    'Inflammation',
    'Proliferative Lesions',
]
lst_compounds=[
    'lomustine',
    'sulindac',
    'mefenamic acid',
    'cyclophosphamide',
    'chloramphenicol',
    'thioridazine',
    'chlorpromazine',
    'indomethacin',
    'chlorpropamide',
    'tolbutamide',
    'nitrofurantoin',
    'famotidine',
    'phenylbutazone',
    'erythromycin ethylsuccinate',
    'acetaminophen',
    'aspirin',
    'diclofenac',
    'glibenclamide',
    'haloperidol',
    'fenofibrate',
    'azathioprine',
    'gemfibrozil',
    'nitrofurazone',
    'carboplatin',
    'ranitidine',
    'sulfasalazine',
    'naproxen']
lst_moa=[
    'Serotonin 2a (5-HT2a) receptor antagonist',
    'Cyclooxygenase inhibitor',
    'Sulfonylurea receptor 1, Kir6.2 blocker',
    'Peroxisome proliferator-activated receptor alpha agonist',
    'Bacterial 70S ribosome inhibitor',
    'DNA inhibitor',
    'Histamine H2 receptor antagonist']

WISTERIA=False
if WISTERIA:
    folder_data="/work/gd43/a97001/data/info"
else:
    folder_data="/workspace/pathology/data"
    
# file names
file_all=f"{folder_data}/tggate_info_ext.csv"
file_classification=f"{folder_data}/processed/finding_converted.csv"
file_prognosis=f"{folder_data}/processed/prognosis.csv"
file_moa=f"{folder_data}/processed/moa.csv"

# for the evaluation of other centers
file_tggate=f"{folder_data}/tggate_info.csv"
file_eisai=f"{folder_data}/eisai_info.csv"
file_shionogi=f"{folder_data}/shionogi_info.csv"
file_our=f"{folder_data}/our_info.csv"
