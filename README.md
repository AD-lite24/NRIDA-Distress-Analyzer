# Distress Analyzer

Code to analyze and evaluate a pothole's volume as well assign it a severity rating

Initialize your environment using

`pip install -r requirements.txt`

or just use the environment.yml file to set up the environment. Ping me if dependencies still give issues. 

The model can be downloaded from the releases section. Modify the path to the file as required

## Files

- The files utils.py and pothole_analysis.py are the part of the integrable module
- The sample_notebook.ipnyb is a testing and sample notebook and is not meant as a final usable file by any means but has only been added to provide additional context if required. However it wouldn't be needed.

NOTE: The detector model for potholes does not work very well and hence no potholes will be detected even after the code is run. This should be fixed soon enough.
