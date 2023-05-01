clear;close all;clc
set(0,'DefaultFigureWindowStyle','docked');
processed_fMRI_path = '/fileserver/alborz/DataRelease2/dhcp_fmri_pipeline/';
%%
% List all subfolders in '/fileserver/alborz/DataRelease2/dhcp_fmri_pipeline'
folders = dir(processed_fMRI_path);
folders_flag = [folders.isdir];
sub_folders = folders(folders_flag);
sub_folder_names = {sub_folders(3:end).name};
len_subfolder_names = length(sub_folder_names);
%%%
% for now it is hardcoded
count_motion_blocks.min_vols_block_lowmo = 775;
count_motion_blocks.min_vols_block_medmo = 500;
%% Create an empty table.
sz_table = [len_subfolder_names,5]; % we 5 columns in the table
varTypes = ["string","string","double","double","double"];
varNames = ["Subject ID","Degree of motion","# of low motion blocks","# of medium motion blocks", "# of high motion blocks"];
T = table('Size',sz_table,'VariableTypes',varTypes,'VariableNames',varNames);
tab_row_ind = 1;
%%
tic;
for ind1 = 1:len_subfolder_names
    % for each subfolder-list and access the paths to the motion params
    % file
    path = strcat(processed_fMRI_path,sub_folder_names{ind1},'/');
    FileList = dir(fullfile(path, '**', '*motion.tsv'));
    l_FileList = length(FileList);
    for ind2 = 1:l_FileList
        params_name = FileList(ind2).name;
        [~,params_filename,~] = fileparts(params_name);
        params = tdfread(strcat(FileList(ind2).folder,'/',params_name));
        FD = params.framewise_displacement;
        l_FD = length(FD);
        count_vols = 1;
        starting_index= 1;
        count_motion_blocks.count_lowmotion_blocks = 0;
        count_motion_blocks.count_medmotion_blocks = 0;
        count_motion_blocks.count_highmotion_blocks = 0;
        if sum(FD < (0.5+eps)) < l_FD  %%% True means some FDs are greater than 0.5
            for ind3 = 1:l_FD
                if FD(ind3) <= 0.5
                    count_vols = count_vols+1;
                else
                    ending_index = ind3;
                    l_block = ending_index - starting_index +1;
                    count_motion_blocks = update_motionblk_count(l_block, count_motion_blocks);
                    if ind3 < l_FD
                        starting_index = ind3+1;
                        count_vols = 1;
                    else
                        starting_index = l_FD;
                    end
                end
            end
            ending_index = l_FD;
            l_block = ending_index - starting_index +1;
            if l_block ~=1
                count_motion_blocks = update_motionblk_count(l_block, count_motion_blocks);
            end
            if count_motion_blocks.count_lowmotion_blocks ~=0
                if count_motion_blocks.count_highmotion_blocks == 0
                    motion_state = "low-medium";
                elseif count_motion_blocks.count_medmotion_blocks == 0
                    motion_state = "low-high";
                else
                    motion_state = "low-medium-high";
                end
            else
                if count_motion_blocks.count_highmotion_blocks == 0
                    motion_state = "medium";
                elseif count_motion_blocks.count_medmotion_blocks == 0
                    motion_state = "high";
                else
                    motion_state = "medium-high";
                end
            end

        else
            motion_state = "very low";
        end
        
        T(tab_row_ind,:) = table(string(params_filename),motion_state,count_motion_blocks.count_lowmotion_blocks,count_motion_blocks.count_medmotion_blocks,count_motion_blocks.count_highmotion_blocks);
        tab_row_ind = tab_row_ind+1;
    end
end
toc;
writetable(T,'/fileserver/fetal/Arvind/fMRI/DHCP/DHCP_subjects_motioninfo.xlsx'); 

function count_motion_blocks = update_motionblk_count(l_block, count_motion_blocks)

min_vols_block_lowmo = count_motion_blocks.min_vols_block_lowmo;
min_vols_block_medmo = count_motion_blocks.min_vols_block_medmo;
if l_block >=min_vols_block_lowmo
    count_motion_blocks.count_lowmotion_blocks =  count_motion_blocks.count_lowmotion_blocks + 1;
elseif l_block >= min_vols_block_medmo && l_block < min_vols_block_lowmo
    count_motion_blocks.count_medmotion_blocks =  count_motion_blocks.count_medmotion_blocks + 1;
else
    count_motion_blocks.count_highmotion_blocks =  count_motion_blocks.count_highmotion_blocks + 1;
end

end



