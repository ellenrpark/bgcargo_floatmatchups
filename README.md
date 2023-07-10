# bgcargo_floatmatchups

From a pandas DataFrame formatted with the following headers:
['LATITUDE', 'LONGITUDE', 'START_DATE', 'END_DATE', 'OUT_FNAME', _d_type_]
where _d_type_ specifies a vertical axis (i.e. 'DEPTH' or 'PRES') find all BGC-Argo floats within a specified distance threshold (in km), _dist_thresh_, measuring the specified list of BGC-ARGO parameters (such as 'DOXY','BBP700'), _params_.

Once these floats are found, profiles are quality-controlled (only use data with QC flag of 1,2). Depth and potential density are calculated. Then, the average values of parameters at specified depth/pressure (_d_type_) +/- a specific value (_d_thresh_) are calculated and saved as a CSV for each match-up location. Quality-controlled BGC-Argo profiles will be saved in a pickle locally and used in future match-ups, unless _over_write_==True.

By default, the program will download the synthetic index file and profiles from the ifremer ('https://data-argo.ifremer.fr/'), so no local copy of the Argo GDAC is needed!

Run mooring_matchups.py to get match-ups for the GOHSNAP mooring.
- Locations, dates, etc. are taken from match_up.csv, which is a reformatted version of the deployment table.
- getfloats_gohsnap.py contains all the main functions
