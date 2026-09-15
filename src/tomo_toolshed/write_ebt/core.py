# Import required libraries
import os
import click

@click.command()
@click.option('--o', prompt='Enter the name of the ebt file', help='Enter the name of the ebt file')
def main(o):

    # Define the text to copy into the ebt file
    header_text = '''
    meta.dataset=true
    meta.EnableStartingStep=false
    meta.dataset.Use.fakeSIRTiterations=false
    meta.dataset.header.open=true
    meta.dataset.fcaleFromZ=0.33
    meta.dataset.Preblend.BinByFactor=1
    meta.ref.ebt.lastID=ebt0
    meta.ProjectLog.FrameLocation.Y=26
    meta.dataset.Use.fcaleFromZ=false
    meta.ProjectLog.FrameLocation.X=26
    meta.dataset.Prenewst.BinByFactor=1
    meta.Version.Etomo.Modified=4.11.21
    meta.ProjectLog.FrameSize.Width=683
    meta.dataset.SizeOfPatchesXandY=680,680
    meta.dataset.autoFitRangeAndStep=true
    meta.dataset.ScaleToInteger=false
    meta.dataset.eraseGold.Fid=true
    meta.dataset.LocalAlignments=false
    meta.datasetTableHeader.open=true
    meta.dataset.LengthOfPieces=false
    meta.dataset.Postprocessing.header.open=false
    meta.dataset.sampleType.Cryo=false
    meta.ProjectLog.Visible=true
    meta.Status=Open
    meta.EndingStep.Use=false
    meta.dataset.hasGoldBeads=true
    meta.StartingStep.Use=false
    meta.Version.Etomo.Created=4.11.21
    meta.dataset.enableStretching=false
    meta.EndingStep=10
    meta.RootName=batchMay08-150740
    meta.dataset.sampleType.PlasticSection=true
    meta.dataset.eraseGold.3d=false
    meta.StartingStep=11
    meta.dataset.Use.findSecAddThickness=false
    meta.dataset.fitEveryImage=false
    meta.ProjectLog.FrameSize.Height=230
    meta.ImageFile.ImageFilenameStyle=MRC
    meta.dataset=true
    meta.datasetTableHeader.open=true
    meta.ref.ebt.lastID=ebt0
    '''

    # Write the text to the ebt file
    with open(o, 'w') as f:
        f.write(header_text)

    # Find all directories ending with "mrc"
    directories = [directory for directory in os.listdir()]

    # Define the starting numerical identifier
    identifier = 0

    # Loop through the directories and add them to the ebt file
    with open(o, 'a') as f:
        for directory in directories:
            identifier += 1
            f.write(f"meta.row.ebt{identifier}.RowNumber={identifier}\n")
            st_path = os.path.abspath(os.path.join(os.getcwd(), directory)) +  "/" + directory + ".st"
#            st_path = st_path.replace("\\","\\\\") 
#            st_path = st_path.replace(":","\:") 
            #st_path = os.path.abspath(os.path.dirname(__file__)) + directory + ".st \n"
            f.write(f"meta.row.ebt{identifier}.OrigStack={st_path}\n")
            f.write(f"meta.row.ebt{identifier}.Tomogram.Done=false\n")
            f.write(f"meta.row.ebt{identifier}.Etomo.Enabled=false\n")
            f.write(f"meta.row.ebt{identifier}.Trimvol.Done=false\n")
            f.write(f"meta.row.ebt{identifier}.Run=true\n")
            f.write(f"meta.row.ebt{identifier}.Rec.Enabled=false\n")
            f.write(f"meta.row.ebt{identifier}.Log.Enabled=false\n")
            f.write(f"meta.ref.ebt{identifier}={st_path}\n")
            f.write(f"meta.row.ebt{identifier}.dual=false\n")
            f.write(f"meta.ref.ebt.lastID=ebt{identifier}\n")
if __name__ == '__main__':
    main()
