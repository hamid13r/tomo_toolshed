import click
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import starfile
import matplotlib.pyplot as plt


@click.command()
@click.option('--star_path', type=click.Path(exists=True), required=True, help='Path to the input STAR file containing particle data.')
@click.option('--output_path', type=str, default=None, help='Path to the output STAR file. If not provided, will use input name with "_cleaned" suffix.')
@click.option('--distance_threshold', type=float, default=140, help='Distance threshold in Angstroms for removing close particles.')
@click.option('--comparison_metric', type=click.Choice(['rlnLogLikeliContribution', 'random']), default='rlnLogLikeliContribution', help='Metric to use for deciding which particle to keep when two are too close.')
def main(star_path, output_path, distance_threshold, comparison_metric):
    """
    This script removes particles that are too close to each other within the same tomogram.
    It iterates through all particles in each tomogram and removes those within the specified
    distance threshold, keeping the particle with the higher log-likelihood contribution (or randomly).
    The process repeats until no more particles need to be removed.
    """
    # Read input STAR file
    input_star = starfile.read(star_path)
    particles_apix = input_star['optics']['rlnImagePixelSize'].values[0]
    print(f"Particles pixel size: {particles_apix} Angstroms")
    print(f"Distance threshold: {distance_threshold} Angstroms")
    print(f"Comparison metric: {comparison_metric}")
    
    tomo_names = input_star['particles']['rlnTomoName'].unique()
    print(f"Number of tomograms: {len(tomo_names)}")
    print(f"Total particles before cleaning: {len(input_star['particles'])}")
    
    cleaned_particles = []
    all_removed_distances = []  # Track distances of all removed particles
    
    for tomo in tomo_names:
        print(f"\nProcessing tomogram: {tomo}")
        tomo_particles = input_star['particles'][input_star['particles']['rlnTomoName'] == tomo].copy()
        initial_count = len(tomo_particles)
        print(f"  Initial particle count: {initial_count}")
        
        # Iteratively remove close particles until no more need to be removed
        iteration = 0
        while True:
            iteration += 1
            print(f"  Iteration {iteration}...")
            
            # Get particle coordinates (considering origins)
            if 'rlnOriginXAngst' in tomo_particles.columns and 'rlnOriginYAngst' in tomo_particles.columns and 'rlnOriginZAngst' in tomo_particles.columns:
                coords = (tomo_particles[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].values * particles_apix 
                         - tomo_particles[['rlnOriginXAngst', 'rlnOriginYAngst', 'rlnOriginZAngst']].values)
            else:
                coords = tomo_particles[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].values * particles_apix
            
            # Build KDTree for efficient distance queries
            tree = cKDTree(coords)
            
            # Find particles to remove
            to_remove = set()
            removed_distances = {}  # Track distances for removed particles {index: distance}
            for i, coord in enumerate(coords):
                if i in to_remove:
                    continue
                
                # find the closest particle
                distances, indices = tree.query(coord, k=2)
                closest_idx = indices[1]
                closest_distance = distances[1]
                #print(f"    Particle {i}: closest distance = {closest_distance:.2f} Å to particle {closest_idx}")
                if closest_distance < distance_threshold:
                    # Decide which particle to keep
                    if comparison_metric == 'rlnLogLikeliContribution':
                        # Keep the particle with higher log-likelihood contribution
                        if tomo_particles.iloc[i]['rlnLogLikeliContribution'] >= tomo_particles.iloc[closest_idx]['rlnLogLikeliContribution']:
                            to_remove.add(closest_idx)
                            removed_distances[closest_idx] = closest_distance
                        else:
                            to_remove.add(i)
                            removed_distances[i] = closest_distance
                    else:  # random
                        # Randomly choose which one to remove
                        if np.random.rand() > 0.5:
                            to_remove.add(closest_idx)
                            removed_distances[closest_idx] = closest_distance
                        else:
                            to_remove.add(i)
                            removed_distances[i] = closest_distance
                            
            print(f"    Particles to remove this iteration: {len(to_remove)}")
            if len(to_remove) == 0:
                print(f"  No more particles to remove. Converged after {iteration} iteration(s).")
                break
            
            print(f"    Removing {len(to_remove)} particles")
            # Store distances of removed particles
            all_removed_distances.extend(removed_distances.values())
            # Remove particles
            tomo_particles = tomo_particles.drop(tomo_particles.index[list(to_remove)]).reset_index(drop=True)
        
        final_count = len(tomo_particles)
        removed_count = initial_count - final_count
        print(f"  Final particle count: {final_count} (removed {removed_count} particles, {removed_count/initial_count*100:.1f}%)")
        
        cleaned_particles.append(tomo_particles)
    
    # Combine all cleaned particles
    all_cleaned = pd.concat(cleaned_particles, ignore_index=True)
    print(f"\nTotal particles after cleaning: {len(all_cleaned)}")
    print(f"Total particles removed: {len(input_star['particles']) - len(all_cleaned)}")
    
    # Create output STAR file
    output_star = {
        'optics': input_star['optics'],
        'particles': all_cleaned
    }
    
    # Determine output path
    if output_path is None:
        output_path = star_path.replace('.star', '_cleaned.star')
    
    # Write output STAR file
    starfile.write(output_star, output_path, overwrite=True)
    print(f"\nCleaned STAR file written to: {output_path}")
    
    # Create histogram of removed particle distances
    if len(all_removed_distances) > 0:
        plt.figure(figsize=(10, 6))
        plt.hist(all_removed_distances, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('Distance between removed particle and its neighbor (Angstroms)', fontsize=12)
        plt.ylabel('Number of removed particles', fontsize=12)
        plt.title(f'Distribution of distances for removed particles\n(Threshold: {distance_threshold} Å, Total removed: {len(all_removed_distances)})', fontsize=14)
        plt.axvline(x=distance_threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({distance_threshold} Å)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save histogram
        histogram_path = output_path.replace('.star', '_removed_distances_histogram.png')
        plt.savefig(histogram_path, dpi=300, bbox_inches='tight')
        print(f"Histogram saved to: {histogram_path}")
        
        # Print statistics
        print(f"\nDistance statistics for removed particles:")
        print(f"  Mean: {np.mean(all_removed_distances):.2f} Å")
        print(f"  Median: {np.median(all_removed_distances):.2f} Å")
        print(f"  Min: {np.min(all_removed_distances):.2f} Å")
        print(f"  Max: {np.max(all_removed_distances):.2f} Å")
        print(f"  Std: {np.std(all_removed_distances):.2f} Å")
    else:
        print("\nNo particles were removed, so no histogram was generated.")


if __name__ == '__main__':
    main()
