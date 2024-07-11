import os
import pandas as pd
import utils


# load data from {project_root}/data/ttc-streetcar-delay-data-2023.xlsx
source_path = os.path.abspath("./data/ttc-streetcar-delay-data-2023.xlsx")
dest_path = os.path.abspath("./data/ttc-streetcar-delay-data-2023-pure.xlsx")
target_cols = ['Date', 'Route', 'Time', 'Day', 'Incident', 'Delay']


dat = utils.purify_xlsx(source_path, dest_path, target_cols)



