import subprocess
import shutil
import os
import time, datetime
import matlab.engine
from median_subtraction import medianSubtraction
import gc
import json


#wait until other day is done
fileExists = False
while not fileExists:
	#directoryToCheck = r'C:\data\processingLogs'
	#fileToCheck = '05162019.txt'

	directoryToCheck = r'D:\05162019\continuous\Neuropix-PXI-slot3-probe3-AP'
	fileToCheck = 'whitening_mat_inv.npy'

	fileExists = os.path.exists(os.path.join(directoryToCheck, fileToCheck))
	time.sleep(30)


########## input block to be modified by user ############
start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
main_start = time.clock()
# select which actions to run
runLocalBackup = True
runRaidBackup = True
runExtraction = True	
runMedianSubtraction = True
runKilosort = True
runCopySortingResults = True

#specify which probes to process
probes_to_run = ['A','B','C','D','E','F']
#probes_to_run = ['E', 'F']

#IF EXTRACTION: npx files to extract. ELSE just give file with correct date.
npx_dirs = [r'E:\2019-05-17_15-14-34_ABC', r'G:\2019-05-17_15-14-34_DEF'] 
npx_raid_backup_base = r'X:\\'
npx_local_backup_base = r'H:\\'
npx_extraction_base = r'D:\\'

########## end of input block #############

def getBackupDirFromNPX(npxdir, backup_base):
	base = os.path.basename(npxdir)
	time_parts = base.split('-')
	year = time_parts[0]
	month = time_parts[1]
	day = time_parts[2][:2]
	full = month+day+year

	candidatePath = os.path.join(backup_base, full)
	if os.path.isdir(candidatePath):
		finalPath = candidatePath
	else:
		dirsInBase = [d for d in os.listdir(backup_base) if os.path.isdir(os.path.join(backup_base, d))]
		possibleDir = [d for d in dirsInBase if full in d]
		if len(possibleDir)>0:
			finalPath = os.path.join(backup_base, possibleDir[0])
		else:
			finalPath = os.path.join(backup_base, full)

	return finalPath

npx_extraction_dir = getBackupDirFromNPX(npx_dirs[0], npx_extraction_base)
npx_raid_backup_dir = getBackupDirFromNPX(npx_dirs[0], npx_raid_backup_base)
npx_local_backup_dir = getBackupDirFromNPX(npx_dirs[0], npx_local_backup_base)

print(npx_extraction_dir)
print(npx_raid_backup_dir)
print(npx_local_backup_dir)

try:
	#dictionary to translate probes to indices
	probeDict = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5}
	probes_to_run_ind = [probeDict[i] for i in probes_to_run]

	# backup to local disk
	if runLocalBackup:
		for npx in npx_dirs:
			command_string = ['robocopy', npx, npx_local_backup_dir]
			print(command_string)
			subprocess.call(command_string)

	# backup to RAID
	if runRaidBackup:
		for npx in npx_dirs:
			command_string = ['robocopy', npx, npx_raid_backup_dir]
			print(command_string)
			subprocess.call(command_string)

	# extract data to binary
	if runExtraction:
		#path to npx extraction exe
		npx_exe_path = r'C:\Users\svc_neuropix\Documents\GitHub\npxextractor\NpxExtractor\Release\NpxExtractor.exe'
		#npx_exe_path = r'C:\Users\svc_neuropix\Documents\GitHub\open-ephys\Tools\NpxExtractor\NpxExtractor.exe'
		if not os.path.isdir(npx_extraction_dir):
				os.mkdir(npx_extraction_dir) #make destination directory
		for npx in npx_dirs:
			command_string = [npx_exe_path, npx, npx_extraction_dir]
			print(command_string)
			subprocess.call(command_string)

	# get paths to binary files
	if runMedianSubtraction or runKilosort or runCopySortingResults:
		datFileRoot = os.path.join(npx_extraction_dir, 'continuous')
		datFileDirs = [f for f in os.listdir(datFileRoot) if (os.path.isdir(os.path.join(datFileRoot,f)) and 'AP' in f)]
		datFileDirs = [datFileDirs[i] for i in probes_to_run_ind]

	#median filter data
	if runMedianSubtraction:
		for datDir in datFileDirs:
			print('Running median subtraction on: ' + os.path.join(datFileRoot, datDir))
			start = time.clock()
			filepath = os.path.join(os.path.join(datFileRoot, datDir), 'continuous.dat')
			command_string = ['python', 'median_subtraction.py', filepath]
			subprocess.check_call(command_string)
			#medianSubtraction(filepath)	
			elapsed = time.clock()-start
			print('Time elapsed (s):' + str(elapsed))
			#gc.collect()


	#run kilosort
	if runKilosort:
		for datDir in datFileDirs:
			print('Running kilosort on: ' + datDir)

			#start matlab engine
			eng = matlab.engine.start_matlab()

			#set root directory with binary
			eng.workspace['rootZ'] = os.path.join(datFileRoot, datDir)
			
			#run kilosort
			eng.config_3B1(nargout=0)
			eng.master_kilosort_3B1(nargout=0)
			eng.quit()

	# copy sorting results to RAID
	if runCopySortingResults:

		for i, datDir in enumerate(datFileDirs):
			absolute_datDir = os.path.join(datFileRoot, datDir)
			filesToCopy = [f for f in os.listdir(absolute_datDir) if os.path.splitext(f)[1] in ('.npy', '.tsv', '.py', '.log', '.mat', '.csv', '.png')]
			dest = os.path.join(npx_raid_backup_dir, os.path.basename(absolute_datDir + '_sortingResults')) #where to copy sorting results
			if not os.path.isdir(dest):
				os.mkdir(dest) #make destination directory
			print('Copying files from: ' + absolute_datDir + ' to: ' + dest)
			command_string = ['robocopy', absolute_datDir, dest, '*.npy', '*.tsv', '*.py', '*.log', '*.mat', '*.csv']
			subprocess.call(command_string)
			# for f in filesToCopy:
			# 	command_string = ['robocopy', absolute_datDir, dest, f]
			# 	subprocess.Popen(command_string)
	errorMSG = 'none'

except Exception as e:
	errorMSG = e

finally:
	main_elapsed = time.clock()-main_start
	print('Total time elapsed (h): ' + str(main_elapsed/3600))

	#write log of completion
	logDirectory = r'C:\data\processingLogs'
	logData = {'raidBackupDir': npx_raid_backup_dir,
				'localBackupDir': npx_local_backup_dir,
				'npxExtractionDir': npx_extraction_dir,
				'startTime': start_time,
				'endTime': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
				'elapsedTime': main_elapsed/3600.,
				'runLocalBackup': runLocalBackup,
				'runRaidBackup': runRaidBackup,
				'runExtraction': runExtraction,	
				'runMedianSubtraction': runMedianSubtraction,
				'runKilosort': runKilosort,
				'runCopySortingResults': runCopySortingResults,
				'error': errorMSG}

	logFileName = os.path.basename(npx_raid_backup_dir)+'.txt'
	with open(os.path.join(logDirectory, logFileName), 'w') as outfile:
		json.dump(logData,outfile)

