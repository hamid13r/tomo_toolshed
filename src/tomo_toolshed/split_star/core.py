import os
import starfile
from collections import defaultdict
import click

@click.command()
@click.option('--i', help='Path to the STAR file')
@click.option('--label', help='added label to the output files')
def split_star(i,label):

    starfile_path = i
    # Read STAR file using starfiles package
    star_data = starfile.read(starfile_path)

    # Group data by micrograph name
    #mic_list = star_data
    mic_list = star_data['rlnMicrographName'].unique()
    #print(mic_list)
    # Process each micrograph group
    for micrograph_name in mic_list:
        # Create directory for each micrograph
        dirname = micrograph_name.replace('.mrc.tomostar', '')
        micrograph_dir = label + '_' + os.path.join(dirname)
        if not os.path.exists(micrograph_dir):
            os.makedirs(micrograph_dir)
        
        # Get rows for the current micrograph
        rows = star_data[star_data['rlnMicrographName'] == micrograph_name]
        #write this micrograph to a star file
        starfile.write(rows, micrograph_dir + '/' + label + '_' + dirname + '_all.star',overwrite=True)
        # separate the rows based on the class number
        #class_list = star_data['rlnClassNumber'].unique()
        #for classnum in class_list:
        #    outstar = rows[rows['rlnClassNumber'] == classnum]
        #    starfile.write(outstar, micrograph_dir + '/' + label + '_' + dirname + '_class_' + str(classnum) + '.star',overwrite=True)



if __name__ == '__main__':
    split_star()