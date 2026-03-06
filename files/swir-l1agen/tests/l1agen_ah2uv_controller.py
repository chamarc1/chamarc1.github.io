"""

__author__ = cjescobar

Copied from same author's "UV_Pushbroom_Gen_Controler.py" code from cjescobar/AH2_UV_pushbroom_gen repo, commit dec4b7ea43 (most recent commit relevant to that script prior to copying)
Modifications made after copying aforementioned script as necessary (*STILL IN PROGRESS)

"""

#----------------------------------------------------------------------------
#-- IMPORT STATEMENTS
#----------------------------------------------------------------------------
# PYTHON STANDARD LIBRARIES
import os;
import sys;
import datetime as dt;
import subprocess;

# REQUIRED ADDITIONAL LIBRARIES/MODULES
import numpy as np;

# CUSTOM MODULES
import modules_l1a_uv as uv1a;


#----------------------------------------------------------------------------
#-- FUNCTION DEFINITIONS
#----------------------------------------------------------------------------
def main():
    args                 = uv1a.argparse_util.parse_l1agen_ah2uv_controller_args();
    uv_input_dir         = args.uv_input_dir;
    nc_output_dir        = args.nc_output_dir;
    int_time             = args.int_time;
    er2_infile           = args.er2_infile;
    ah2_imu_ncfile       = args.ah2_imu_ncfile;
    shell_output_fname   = args.shell_output_fname;
    shell_output_dir     = args.shell_output_dir;
    shell_log_output_dir = args.shell_log_output_dir;
    full_flight_leg_flag = args.full_flight_leg_flag;
    num_imgs             = args.num_imgs;
    time_tolerance_sec   = args.time_tolerance_sec;
    disable_autorun      = args.disable_autorun;
    
    
    #-- get directory from this script instance's absolute path; using relative paths for other calls may fail or result in unintended behavior if this script's absolute directory is not the user's current working directory; e.g. if the user calls this script using its absolute path while the user is in their home directory, then any relative paths will default to the user's home directory instead
    controller_dir = os.path.dirname(os.path.abspath(__file__));
    
    
    #-- determine start_img and end_img arguments for l1agen_ah2uv.py Python script, where each entry in start_img_list, end_img_list indicates the start and end of each pushbroom
    all_img_files = os.listdir(uv_input_dir);
    all_img_files.sort();
    prev_timestamp = dt.datetime.strptime(all_img_files[0].split("_")[0], "%Y%m%d%H%M%S");  # initialize reference timestamp for chronological checking of flight legs
    prev_idx = 0;
    idx_bounded_files = np.asarray([]);  # initialize array of acquisition-time-bounded UV image files' indices, starting with chronological first image in directory. format will be applied during loop iteration for populating it: [start_end_pair][start_index][end_index]
    img_counter = 0;  # initialize counter for number of images, to compare against num_imgs

    # get UV images that fall within acquisition time bounds
    for fname in all_img_files:
        if fname.lower().endswith(".db"):
            continue;  # skip thumbnail files; move to next loop iteration
        
        timestamp = dt.datetime.strptime(fname.split("_")[0], "%Y%m%d%H%M%S");
        idx = all_img_files.index(fname);
        
        # number of images per NC only matters if flight legs are to be divided instead of saved as full legs; see full_flight_leg_flag global
        if not full_flight_leg_flag:
            img_counter += 1;
        # end if
        
        # append index of current image in directory, which indicates that a new flight leg and/or continuous period of acquisition has started, based on time difference from the previous image. i.e. if the difference between current image and previous image is greater than time_tolerance_sec, then enough time has elapsed between those images to indicate that acquisition ended at the previous image and acquisition started again at current image. Appending index - 1 so that start image in next loop iteration is current iteration's index
        if (timestamp - prev_timestamp) > dt.timedelta(seconds = time_tolerance_sec):
            if idx_bounded_files.shape == (0,):
                idx_bounded_files = np.asarray([prev_idx, idx - 1]);  # set idx_bounded_files == first pair that meets condition so that its shape is properly set for vstacking remaining pairs
                img_counter = 0;
            else:
                idx_bounded_files = np.vstack((idx_bounded_files, [prev_idx, idx - 1]));
                img_counter = 0;
            prev_idx = idx; # set prev_idx to current index to prepare for next loop iteration. MUST occur in if statement so that prev_idx is ONLY overwritten when a valid start/end index has been met
            # end if
        # append final pair of start, end times where end is final image in directory
        elif img_counter == num_imgs:
            idx_bounded_files = np.vstack((idx_bounded_files, [prev_idx, idx - 1]));
            prev_idx = idx;
            img_counter = 0;
        elif idx == len(all_img_files) - 1:
            idx_bounded_files = np.vstack((idx_bounded_files, [prev_idx, idx]));
        # end if
        prev_timestamp = timestamp;  # set prev_timestamp to current timestamp to prepare for next loop iteration 
    # end for
    
    start_img_list = np.asarray(all_img_files)[idx_bounded_files.T[0]];
    end_img_list   = np.asarray(all_img_files)[idx_bounded_files.T[1]];


    #-- check if Slurm is installed; if it is, then Slurm will be used for parallel processing
    try:
        slurm_check      = subprocess.run(["sinfo"], check = True, stdout = subprocess.DEVNULL, stderr = subprocess.DEVNULL);
        slurm_installed  = True;
        parallel_command = "srun --ntasks=1 --mem-per-cpu=2G --exclusive";
    except (FileNotFoundError, subprocess.CalledProcessError):
        slurm_installed  = False;
        parallel_command = f"parallel -j {uv1a.constants.NTHREADS_MAX}";
    # end try
    
    
    #-- determine l1agen_ah2uv.py command string, using absolute path of l1agen_ah2uv.py from local repo (l1agen_ah2uv.py MUST be in the same directory as l1agen_ah2uv_controller.py for this to work, since the path is determined by finding the absolute path to this l1agen_ah2uv_controller.py instance and using that same directory's instance of l1agen_ah2uv.py)
    py_command_call = f"python {os.path.join(controller_dir, 'l1agen_ah2uv.py')}";
    
    
    #-- determine argument strings common to all granule calls for given flight day
    io_dirs_args = f'''--uv_input_dir "{uv_input_dir}" --nc_output_dir "{nc_output_dir}"''';  # UV images dir input, L1A NetCDF4 dir output args
    int_time_arg = f'''--int_time {int_time}''';  # integration time arg
    
    # IMU args
    if (ah2_imu_ncfile is not None) and (ah2_imu_ncfile.lower() != "none") and (os.path.exists(ah2_imu_ncfile)):
        ah2_imu_arg = f'''--ah2_imu_ncfile "{ah2_imu_ncfile}"''';
    else:
        ah2_imu_arg = None;
    # end if
    
    if (er2_infile is not None) and (er2_infile.lower() != "none") and (os.path.exists(er2_infile)):
        er2_imu_arg = f'''--er2_infile "{er2_infile}"''';
    else:
        er2_imu_arg = None;
    # end if
    
    common_args = " ".join(filter(None, [io_dirs_args, int_time_arg, ah2_imu_arg, er2_imu_arg]));  # combined string of all common args


    #-- create Shell script and log output directory ONLY IF no directory was specified/default directory is chosen AND if default directory does not exist
    # NOTE: unsure if "runs" folder name should be case-sensitive or not... gitignore currently only ignores fully lowercase "runs"
    if shell_output_dir == "./runs":
        shell_output_dir = os.path.join(controller_dir, "runs"); # using controller_dir so that absolute path of default "runs" directory is in user's local repo; using os.mkdir(shell_output_dir) would otherwise result in making a "runs" dir from the current working directory even if the cwd is not the local repo
        
        if not os.path.exists(shell_output_dir):
            os.mkdir(shell_output_dir);  
        # end if
    # end if
    
    # !!! NEED TO MODIFY CONFIG TEMPLATE BASED ON %u ONLY BEING APPLICABLE TO SLURM !!!
    if shell_log_output_dir == "./runs/log-%u":
        shell_log_output_dir = os.path.join(controller_dir, f"runs/log-{os.getenv('USER')}");  # NOTE: "%u" is Slurm's placeholder for username
    # end if

    if (not os.path.exists(shell_log_output_dir)) and ((shell_log_output_dir == os.path.join(controller_dir, "runs/log-%u")) or (shell_log_output_dir == os.path.join(controller_dir, f"runs/log-{os.getenv('USER')}"))):
        os.mkdir(shell_log_output_dir);  
    # end if


    #-- generate Slurm job/GNU parallel Shell script. note that ntasks = number of tasks in Slurm job to run in parallel, which in this case is determined by number of NC files to generate, which each NC file containing pushbrooms from num_imgs
    with open(os.path.join(shell_output_dir, shell_output_fname), "w") as output_sh:
        output_sh.write(r'''#!/bin/bash''');
        output_sh.write("\n\n");
        if slurm_installed:
            output_sh.write(f'''#SBATCH --job-name=L1AGen_AH2UV
#SBATCH --ntasks={len(start_img_list)}
#SBATCH --partition=zen4
#SBATCH --nodelist=tiger-0[1-3]
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --output={shell_log_output_dir}/%j.out
#SBATCH --error={shell_log_output_dir}/%j.err''');
        else:
            output_sh.write(f"exec 1>{shell_log_output_dir}/$$.out");
            output_sh.write("\n");
            output_sh.write(f"exec 2>{shell_log_output_dir}/$$.err");
            output_sh.write("\n");
        # end if
        
        output_sh.write('\n\nSTART_TIME=$(date +"%s")  # seconds elapsed since Unix epoch');
        output_sh.write("\necho JOB START TIME: $(date -d @$START_TIME)");
        output_sh.write("\necho PYTHON STDOUT STARTS HERE-----------------");
        output_sh.write("\necho")
        output_sh.write("\n\n");
        
        # line for Python venv activation; venv for main driver calls should be same as venv for this controller script, so getting venv activation based on venv being used at controller runtime
        if os.getenv("VIRTUAL_ENV") is not None:  # pip
            output_sh.write(f'''source "{os.path.join(os.getenv("VIRTUAL_ENV"), "bin/activate")}" # path to L1A AH2UV venv activation''');
        else:  # conda; "VIRTUAL_ENV" environment variable is None for conda environments
            output_sh.write(f'''conda activate {os.path.basename(sys.prefix)}  # path to L1A AH2UV venv activation''');
        # end if
        
        output_sh.write("\n");
        
        # parallel main driver processes called via Slurm
        if slurm_installed:
            for start_img, end_img in zip(start_img_list, end_img_list):
                output_sh.write("\n");
                
                command_str = " ".join(filter(None, [parallel_command, py_command_call, common_args, f'''--start_img "{start_img}" --end_img "{end_img}"''', "-v &"]));

                output_sh.write(command_str);
            # end for
        
            output_sh.write("\nwait\n");
        
        # parallel main driver processes called via GNU parallel
        else:
            output_sh.write("\n");
            
            output_sh.write(f'''START_IMG_LIST=({" ".join("'" + start_img + "'" for start_img in start_img_list)})''');  # bash script variables must NOT have spaces surrounding "="; using concatenation to manually add quotes so that fully joined string contains substrings, allowing for Python list of strings to be properly represented as Bash array of strings
            output_sh.write("\n\n");
            output_sh.write(f'''END_IMG_LIST=({" ".join("'" + end_img + "'" for end_img in end_img_list)})''');  # bash script variables must NOT have spaces surrounding "="; using concatenation to manually add quotes so that fully joined string contains substrings, allowing for Python list of strings to be properly represented as Bash array of strings
            output_sh.write("\n\n");
            
            command_str = " ".join(filter(None, [parallel_command, py_command_call, common_args, "--start_img {1} --end_img {2} -v", "::: ${START_IMG_LIST[@]} :::+ ${END_IMG_LIST[@]}"]));  # :::+ links second argument list to first argument list so that instead of making all possible combinations, e.g. start_img1 end_img1 start_img1 end_img2, arguments are properly paired, e.g. start_img1 end_img1 start_img2 end_img2
            
            output_sh.write(command_str);
            output_sh.write("\n");
        
        # end if
        
        output_sh.write("\necho");
        output_sh.write("\necho PYTHON STDOUT ENDS HERE-----------------");
        output_sh.write('\nEND_TIME=$(date +"%s")  # seconds elapsed since Unix epoch');
        output_sh.write("\necho JOB END TIME: $(date -d @$END_TIME)");
        output_sh.write("\necho");
        output_sh.write("\nDURATION=$(( END_TIME - START_TIME ))  # full time elapsed in seconds");
        output_sh.write('\nDURATION_DD=$(( DURATION / 86400 ))  # num days elapsed; HH:MM:SS determined by "date" command');
        output_sh.write('\necho JOB TIME ELAPSED "(DD-HH:MM:SS)": $DURATION_DD-$(date -d @$DURATION -u +%T)');
        output_sh.close();
    # end with


    #-- submit job to Slurm or otherwise run Shell script with baked-in GNU parallel
    if not disable_autorun:
        if slurm_installed:
            subprocess.run(["sbatch", f"{os.path.join(shell_output_dir, shell_output_fname)}"]);
        else:
            called_subprocess = subprocess.Popen(["nohup", f"{os.path.join(shell_output_dir, shell_output_fname)}", "&"], stdout = subprocess.DEVNULL, stderr = subprocess.DEVNULL);  # nohup will try to write to nohup.out, so redirect stdout and stderr to /dev/null; Shell script itself has lines for properly redirecting stdout and stderr to log files named by PID
            print(f"{os.path.join(shell_output_dir, shell_output_fname)} running in background with PID: {called_subprocess.pid}");
            print(f"Check job status using the following command: 'ps -p {called_subprocess.pid}'");
        # end if
    # end if


if __name__ == "__main__":
    main();
