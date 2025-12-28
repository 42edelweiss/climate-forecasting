import numpy as np

from raw_time_series import raw_time_series

ts= raw_time_series()

print(f'Count= {len(ts)}')

print(f'Min :{np.nanmin(ts)}')

print (f'Avg:{round (np.nanmean(ts), 2)}')

print (f'Max:{np.nanmax(ts)}')

print (f'Std Dev:{round (np.nanstd(ts), 2)}')

print (f'Median:{np.nanmedian(ts)}')

print (f'NA values :{np.count_nonzero(np.isnan(ts))}')
