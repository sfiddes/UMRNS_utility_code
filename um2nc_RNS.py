import sys
import os
import numpy as np 
from datetime import datetime
import xarray as xr
import iris
import argparse
import cartopy.crs as ccrs
from iris.analysis.cartography import unrotate_pole, rotate_pole
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="File names to read and write")
    parser.add_argument("fname", type=str, help="Name of the file to read")
    parser.add_argument("fout", type=str, help="Name of file to save")
    parser.add_argument("clon", type=float, help="central longitude of rotated grid")
    parser.add_argument("clat", type=float, help="central latitude of rotated grid")
    parser.add_argument("extract_point", type=str, default='False', help="extract data at single point, if no target_lat target_lon provided, will use central point")
    #Assuming that the chances of 0,0degrees lat lon as a central point is very low, setting defaults to 0,0 
    parser.add_argument("target_lon", type=float, nargs='?', default=0, help="Target longitude for point extraction, defaults to central longitude if not provided")
    parser.add_argument("target_lat", type=float, nargs='?', default=0, help="Target latitude for point extraction, defaults to central latitude if not provided")
    parser.add_argument("extract_ship", type=str, nargs='?', default='None', help="Extract along a ship track, default is 'None'. Compatible shiptracks: MISO")
    args = parser.parse_args()

    fname = args.fname
    fout = args.fout
    clon = args.clon
    clat = args.clat
    extract_point = args.extract_point
    target_lon = args.target_lon
    target_lat = args.target_lat
    extract_ship = args.extract_ship
    
    if extract_point=='True':     
        if target_lon == 0: 
            target_lon = args.clon
         
        if target_lat == 0: 
            target_lat = args.clat

    #print(fname,fout,clon,clat,extract_point,target_lon,target_lat,extract_ship)
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

        if data.attrs['STASH'].item == 96 and data.attrs['STASH'].section == 0: 
            data = data.rename('ocean_near_surface_chlorophyll_km_per_m3')  
        if data.attrs['STASH'].item == 437 and data.attrs['STASH'].section == 38: 
            data = data.rename('condensation_nuclei_number_concentration_greater_than_3nm_dry_diameter_per_cm3')  
        if data.attrs['STASH'].item == 439 and data.attrs['STASH'].section == 38: 
            data = data.rename('CCN_greater_than_50nm_dry_diameter_AIT+ACC+COA_per_cm3') 

        if data.attrs['STASH'].item == 230 and data.attrs['STASH'].section == 3: 
            data = data.rename({'height':'height1'})  
        if data.attrs['STASH'].item == 209 and data.attrs['STASH'].section == 3: 
            data = data.rename({'height':'height1'}) 
        if data.attrs['STASH'].item == 210 and data.attrs['STASH'].section == 3: 
            data = data.rename({'height':'height1'}) 

        # Add some compression info for writing: 
        # This takes foooorrreeevverrrr.
        #comp = dict(zlib=True, complevel=7)
        #data.encoding.update(comp) 
        
        dataout = dataout.merge(data)
        
    dataout = dataout.assign_attrs({'central latitude of rotated grid':clat,
                                'central longitude of rotated grid':clon,
                                'history':'Data generated and processed by S. Fiddes sonya.fiddes@utas.edu.au {}'.format(datetime.today().date())})
    
    
    dataout.to_netcdf(fout)

    #*************************************

    # Extract at a point
    if extract_point=='True':
        print('Extracting at point {}, {}'.format(target_lon,target_lat))
    
        grid_lon, grid_lat = rotate_pole(np.array(target_lon), np.array(target_lat), clon, clat+90)
        grid_lon = grid_lon % 360
        point_data = dataout.copy()
        if 'grid_latitude' in dataout.dims: 
            point_data = point_data.sel(grid_latitude = grid_lat, grid_longitude = grid_lon, method='nearest')
        if 'grid_latitude1' in dataout.dims: 
            point_data = point_data.sel(grid_latitude1 = grid_lat, grid_longitude1 = grid_lon, method='nearest')
        if 'grid_latitude2' in dataout.dims: 
            point_data = point_data.sel(grid_latitude2 = grid_lat, grid_longitude2 = grid_lon, method='nearest')
        
        point_data.to_netcdf(fout[:-3]+'_centralpoint'+fout[-3:])

    #*************************************

    # Extract along a ship track
    if extract_ship!='None': 
        print('extracting data along '+extract_ship+' shiptrack') 
        if extract_ship == 'MISO':
            # This is the 1 min underway data - hopefully all RVI underway data is similar in structure
            shiptrack = pd.read_csv('/g/data/jk72/slf563/OBS/campaigns/MISO/IN2024_V01_uwy_data.csv',index_col=0)
            shiptrack.index = pd.DatetimeIndex(shiptrack.index)
            shiptrack = shiptrack.drop(shiptrack[shiptrack['Latitude (degree_north)']>-62.505].index)
            shiptrack = shiptrack.drop(shiptrack[shiptrack['Longitude (degree_east)']<137.3].index)
            shiptrack = shiptrack.drop(shiptrack[shiptrack['Longitude (degree_east)']>145.4].index)
            
        #elif extract_ship == 'new ship track': 
            
        else: 
            'Ship track not recognised - add code above for desired ship track'
        
        time = pd.DatetimeIndex(dataout.time) # convert model time to datetimeindex to loop through and 
    
        ship_data = xr.Dataset()
        for t in time: 
            if t > shiptrack.index[0] and t < shiptrack.index[-1]:
                print('voyage data found for this time range')
    
                lat = shiptrack.loc['{}-{:02}-{} {:02}{:02}'.format(
                    t.year,t.month,t.day,t.hour,t.minute)]['Latitude (degree_north)']  
                lon = shiptrack.loc['{}-{:02}-{} {:02}{:02}'.format(
                    t.year,t.month,t.day,t.hour,t.minute)]['Longitude (degree_east)']
                
                grid_lon, grid_lat = rotate_pole(lon, lat, clon, clat+90)
                grid_lon = grid_lon % 360
                
                point = dataout.sel(time=t)
                if 'time1' in dataout.dims:
                    point = point.sel(time1=t,method='nearest')
                    point = point.drop_vars('time1')
                    point = point.drop_vars('forecast_period1')
                if 'grid_latitude' in dataout.dims:
                    point = point.sel(grid_latitude = grid_lat, grid_longitude = grid_lon, method='nearest')
                if 'grid_latitude1' in dataout.dims:
                    point = point.sel(grid_latitude1 = grid_lat, grid_longitude1 = grid_lon, method='nearest')
                if 'grid_latitude2' in dataout.dims:
                    point = point.sel(grid_latitude2 = grid_lat, grid_longitude2 = grid_lon, method='nearest')
    
                reformatted_point = xr.Dataset()
                for key in list(point.keys()):
                    tmp = point[key]
                    if 'grid_latitude' in point[key].dims: 
                        tmp = tmp.isel(grid_latitude=0,grid_longitude=0)
                    if 'grid_latitude1' in point[key].dims: 
                        tmp = tmp.isel(grid_latitude1=0,grid_longitude1=0)
                    if 'grid_latitude2' in point[key].dims: 
                        tmp = tmp.isel(grid_latitude2=0,grid_longitude2=0)
                            
                    tmp = tmp.expand_dims('time')
                    reformatted_point[key] = tmp
                    
                point = reformatted_point
        
                # Drop the current grid lat/lons because they are a bit useless and convlute things
                if 'grid_latitude' in list(point.coords):
                    point = point.drop_vars(['grid_latitude','grid_longitude'])
                if 'grid_latitude1' in list(point.coords):
                    point = point.drop_vars(['grid_latitude1','grid_longitude1'])
                if 'grid_latitude2' in list(point.coords):
                    point = point.drop_vars(['grid_latitude2','grid_longitude2'])
        
                # but then we need to include the grid lat/lon as a dimension for the aerosol conversions to work, so 
                # add in as a dummy dim, with only 1 point which equals 0 for all.. 
                point = point.expand_dims('grid_latitude')
                point = point.expand_dims('grid_longitude')
    
                # rearrange dims to be time, level, lat, lon
                if ('model_level_number' in point.dims) and ('model_level_number1' in point.dims):
                    point = point.transpose('time','model_level_number','model_level_number1','grid_latitude','grid_longitude')
                elif 'model_level_number' in point.dims: 
                    point = point.transpose('time','model_level_number','grid_latitude','grid_longitude')
                else: 
                    point = point.transpose('time','grid_latitude','grid_longitude')

                # Add ship location as coordinate...
                point = point.assign_coords({'ship_latitude':lat})
                point['ship_latitude'] = point['ship_latitude'].ship_latitude.expand_dims('time')
                point = point.assign_coords({'ship_longitude':lon})
                point['ship_longitude'] = point['ship_longitude'].ship_longitude.expand_dims('time')
                point = point.drop_vars('forecast_period')
                
                ship_data = ship_data.merge(point)
        
        if len(ship_data)>0:
            ship_data.to_netcdf('{}_{}{}'.format(fout[:-3],extract_ship,fout[-3:]))

if __name__ == "__main__":
    main()
