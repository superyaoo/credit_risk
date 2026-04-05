#!/bin/bash

work_dir=/home/model/
featurelib_path=/home//mahaocheng/workspace/featurelib/
python_bin=/home/mahaocheng/app/miniconda3/envs/feature_lib/bin/

stage=1
country=af
usr=mahaocheng
passwd=tyko6FqcL0mJ0I+w
dblink=inner
start_date=2024-06-01
end_date=2025-08-25
user_type=old
write_mode=incr
max_workers=4
port=10081
config_path=config/af_woe.yaml


if [ $stage -eq 0 ]; then
    python get_loan_req_data.py \
        --country $country \
        --user $usr \
        --passwd $passwd \
        --dblink $dblink \
        --start_date $start_date \
        --end_date $end_date \
        --write_mode $write_mode \
	--user_type $user_type \
        --work_dir $work_dir
fi

if [ $stage -eq 1 ]; then
    python feature_backtracking.py \
        --country $country \
        --config_path $config_path \
        --max_workers $max_workers \
        --port $port \
        --python_bin $python_bin \
        --featurelib_path $featurelib_path \
        --start_date $start_date \
        --end_date $end_date \
        --write_mode $write_mode \
	--user_type $user_type \
        --work_dir $work_dir
fi
