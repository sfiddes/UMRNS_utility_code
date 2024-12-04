Code to process the UM Regional Nesting Suite. 

The notebooks are used to develope the code, which I have then exported to a python script to run via the command line. 

- convert_latlon_coords.ipynb
Notebook  used to test the conversion of the polar rotated grid onto a lat/lon grid 

- convert_um_files_to_nc.ipynb 
Notebook used to develop the code to process the UM files to netcdf format, including renameing some fields to sensible names, converting the lat lon grid and pulling out a central lat/lon point 

- Aerosol_conversions_size_dists.ipynb
Notebook used to develop the code needed to do the aerosol/chemical unit converstions and the size distribution processing 

- um2nc_RNS.py
Does the initial lat/lon, netcdf conversions, is not specific to file names 

- aerosol_converstions.py 
Does the aerosol unit conversions, size distributions, etc. Does depend on files names.  

- simple_iris_wrap_RNS.sh
Runs both the um2nc_RNS.py and aerosol_converstions.py 


