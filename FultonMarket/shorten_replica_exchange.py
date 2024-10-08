import numpy as np
import netCDF4 as nc
from openmmtools.multistate import MultiStateReporter

def truncate_ncdf(ncdf_in, ncdf_out, reporter, is_checkpoint: bool=False):
    print(f'Truncating {ncdf_in} to {ncdf_out}')

    src = nc.Dataset(ncdf_in, 'r')
    dest = nc.Dataset(ncdf_out, 'w')
                      
    for name in src.ncattrs():
        dest.setncattr(name, src.getncattr(name))
    
    for dim_name, dim in src.dimensions.items():
        dest.createDimension(dim_name, (len(dim) if not dim.isunlimited() else None))
    
    for group_name, group in src.groups.items():
        group = dest.createGroup(group_name)
        for name, variable in src[group_name].variables.items():
            try:
                dest[group_name].createVariable(name, variable.datatype, variable.dimensions)
                dest[group_name][name][:] = src[group_name][name][:]
                dest[group_name][name].setncatts(src[group_name][name].__dict__)
            except:
                print(group_name, name)
                pass
    
    for var_name, var in src.variables.items():
        var_out = dest.createVariable(var_name, var.datatype, var.dimensions)
        var_out.setncatts({k: var.getncattr(k) for k in var.ncattrs()})
    
        if not is_checkpoint:
            if var_name == 'positions':
                pos = var[:].copy().astype('float16')
            elif var_name == 'box_vectors':
                box_vecs = var[:].copy()
            elif var_name == 'states':
                states = var[:].copy()
            elif var_name == 'energies':
                energies = var[:].copy().astype('float32')
            elif var_name == 'velocities':
                velocities = var[-1].copy().astype('float16')
        
        if var.dimensions[0] == 'iteration':
            if is_checkpoint:
                var_out[:] = var[-1:]
            else:
                var_out[:] = var[-10:]

        elif var_name == 'last_iteration':
            var_out[:] = var[:]
            if is_checkpoint == False:
                mask_copy = var_out[:].copy()
                var_out[:] = np.ma.array(9, mask=mask_copy.mask, fill_value=mask_copy.fill_value)
                print(var_out)
            
        else:
            var_out[:] = var[:]

    dest.close()    
    src.close()

    # Read reporter
    if not is_checkpoint:

        # Read temperatures
        temps = np.array([state.temperature._value for state in reporter.read_thermodynamic_states()[0]])
        
        # Close reporter
        reporter.close()

        return pos, velocities, box_vecs, states, energies, temps
        



