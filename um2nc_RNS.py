import sys
import os
import numpy as np 
from datetime import datetime
import xarray as xr
import iris
import argparse
import cartopy.crs as ccrs
from iris.analysis.cartography import unrotate_pole, rotate_pole

def main():
    parser = argparse.ArgumentParser(description="Read in a file name from the command line")
    parser.add_argument("fname", type=str, help="Name of the file to read")
    parser.add_argument("fout", type=str, help="Name of file to save")
    parser.add_argument("clon", type=float, help="central longitude of rotated grid")
    parser.add_argument("clat", type=float, help="central latitude of rotated grid")
    parser.add_argument("extract_point", type=bool, default=False, help="extract data at single point, if no target_lat target_lon provided, will use central point")
    #Assuming that the chances of 0,0degrees lat lon as a central point is very low, setting defaults to 0,0 
    parser.add_argument("target_lon", type=float, nargs='?', default=0, help="target longitude for point extraction, defaults to central longitude if not provided")
    parser.add_argument("target_lat", type=float, nargs='?', default=0, help="target latitude for point extraction, defaults to central latitude if not provided")
    args = parser.parse_args()

    fname = args.fname
    fout = args.fout
    clon = args.clon
    clat = args.clat
    extract_point = args.extract_point
    target_lon = args.target_lon
    target_lat = args.target_lat
    
    if extract_point==True:     
        if target_lon == 0: 
            target_lon = args.clon
         
        if target_lat == 0: 
            target_lat = args.clat

    #*************************************
    
    cubes = iris.load(fname)
    dataout = xr.Dataset()

    for i,cube in enumerate(cubes):  
        
        rotated_lons, rotated_lats = cube.coord('grid_longitude').points, cube.coord('grid_latitude').points
        x, y = np.meshgrid(rotated_lons, rotated_lats)
        lons, lats = unrotate_pole(x, y, clon, clat+90)
        
        data = xr.DataArray.from_iris(cube)
        data = data.assign_coords({'latitude':(('grid_latitude','grid_longitude'),lats)})
        data = data.assign_coords({'longitude':(('grid_latitude','grid_longitude'),lons)})
        
        if i > 0 and data.grid_latitude[0].values != dataout.grid_latitude[0].values:
            data = data.rename({'grid_latitude':'grid_latitude1'})
            data = data.rename({'grid_longitude':'grid_longitude1'})
            data = data.rename({'latitude':'latitude1'})
            data = data.rename({'longitude':'longitude1'})
    
        if 'grid_longitude' in data.dims and i > 0 and data.grid_longitude[0].values != dataout.grid_longitude[0].values:
            data = data.rename({'grid_longitude':'grid_longitude2'})
            data = data.rename({'grid_latitude':'grid_latitude2'})
            data = data.rename({'latitude':'latitude2'})
            data = data.rename({'longitude':'longitude2'})

        if 'time' in list(data.coords) and 'time' in list(dataout.coords):    
            if data.time[0].values != dataout.time[0].values:
                data = data.rename({'time':'time1'})
                data = data.rename({'forecast_period':'forecast_period1'})
            
    
        if 'level_height' in list(data.coords) and 'level_height' in list(dataout.coords):    
            if data.level_height[0].values != dataout.level_height[0].values:
                data = data.rename({'level_height':'level_height1'})
                data = data.rename({'model_level_number':'model_level_number1'})

        if 'sigma' in list(data.coords) and 'sigma' in list(dataout.coords):
            if data.sigma[0].values != dataout.sigma[0].values:
                data = data.rename({'sigma':'sigma1'})

        if (data.attrs['STASH'].item == 229) and (data.attrs['STASH'].section == 15): 
            data = data.rename('potential_vorticity')
        if data.attrs['STASH'].item == 293 and data.attrs['STASH'].section == 30:     
            data = data.rename('w_component_of_wind')
        if data.attrs['STASH'].item == 294 and data.attrs['STASH'].section == 30: 
            data = data.rename('temperature')
        if data.attrs['STASH'].item == 295 and data.attrs['STASH'].section == 30:     
            data = data.rename('specific_humidity')
        if data.attrs['STASH'].item == 296 and data.attrs['STASH'].section == 30: 
            data = data.rename('relative_humidity')
        if data.attrs['STASH'].item == 297 and data.attrs['STASH'].section == 30: 
            data = data.rename('geopotential_height')
        if data.attrs['STASH'].item == 304 and data.attrs['STASH'].section == 30: 
            data = data.rename('heavyside_function')  

        if data.attrs['STASH'].item == 401 and data.attrs['STASH'].section == 38: 
            data = data.rename('dry_particle_diameter_soluble_nucleation_mode_aerosol')  
        if data.attrs['STASH'].item == 402 and data.attrs['STASH'].section == 38: 
            data = data.rename('dry_particle_diameter_soluble_aitken_mode_aerosol')  
        if data.attrs['STASH'].item == 403 and data.attrs['STASH'].section == 38: 
            data = data.rename('dry_particle_diameter_soluble_accumulation_mode_aerosol')  
        if data.attrs['STASH'].item == 404 and data.attrs['STASH'].section == 38: 
            data = data.rename('dry_particle_diameter_soluble_course_mode_aerosol')  
        if data.attrs['STASH'].item == 405 and data.attrs['STASH'].section == 38: 
            data = data.rename('dry_particle_diameter_insoluble_aitken_mode_aerosol')  
        if data.attrs['STASH'].item == 406 and data.attrs['STASH'].section == 38: 
            data = data.rename('dry_particle_diameter_insoluble_aitken_mode_aerosol')  
        if data.attrs['STASH'].item == 407 and data.attrs['STASH'].section == 38: 
            data = data.rename('dry_particle_diameter_insoluble_aitken_mode_aerosol')   

        if data.attrs['STASH'].item == 230 and data.attrs['STASH'].section == 3: 
            data = data.rename({'height':'height1'})  
        if data.attrs['STASH'].item == 209 and data.attrs['STASH'].section == 3: 
            data = data.rename({'height':'height1'}) 
        if data.attrs['STASH'].item == 210 and data.attrs['STASH'].section == 3: 
            data = data.rename({'height':'height1'})    


        if data.attrs['STASH'].item == 230 and data.attrs['STASH'].section == 3: 
            data = data.rename({'height':'height1'})  
        if data.attrs['STASH'].item == 209 and data.attrs['STASH'].section == 3: 
            data = data.rename({'height':'height1'}) 
        if data.attrs['STASH'].item == 210 and data.attrs['STASH'].section == 3: 
            data = data.rename({'height':'height1'}) 

        dataout = dataout.merge(data)
        
    dataout = dataout.assign_attrs({'central latitude of rotated grid':clat,
                                'central longitude of rotated grid':clon,
                                'history':'Data generated and processed by S. Fiddes sonya.fiddes@utas.edu.au {}'.format(datetime.today().date())})
            
    dataout.to_netcdf(fout)

    if extract_point==True: 
    
        grid_lon, grid_lat = rotate_pole(np.array(target_lon), np.array(target_lat), clon, clat+90)
        point_data = dataout.copy()
        if 'grid_latitude' in dataout.dims: 
            point_data = point_data.sel(grid_latitude = grid_lat, grid_longitude = grid_lon, method='nearest')
        if 'grid_latitude1' in dataout.dims: 
            point_data = point_data.sel(grid_latitude1 = grid_lat, grid_longitude1 = grid_lon, method='nearest')
        if 'grid_latitude2' in dataout.dims: 
            point_data = point_data.sel(grid_latitude2 = grid_lat, grid_longitude2 = grid_lon, method='nearest')
        
        point_data.to_netcdf(fout[:-3]+'_centralpoint'+fout[-3:])

if __name__ == "__main__":
    main()
