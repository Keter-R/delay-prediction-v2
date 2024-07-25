import os
import pandas as pd
import utils


# load data from {project_root}/data/ttc-streetcar-delay-data-2023.xlsx
source_path = os.path.abspath("./data/ttc-streetcar-delay-data-2023.xlsx")
dest_path = os.path.abspath("./data/ttc-streetcar-delay-data-2023-pure.xlsx")
target_cols = ['Date', 'Route', 'Time', 'Day', 'Incident', 'Delay']
target_routes = ['501', '503', '504', '505', '506', '507', '508', '509', '510', '511', '512', '301', '304', '306', '310']
# target_routes = None
dat = utils.purify_xlsx(source_path, dest_path, target_cols, target_routes)



