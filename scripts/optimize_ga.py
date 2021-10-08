from syn_net.utils.ga_utils import crossover, mutation
import multiprocessing as mp
import numpy as np
import pandas as pd
import time
import json
import scripts._mp_decode as decode
# import scripts._mp_sum as decode
from tdc import Oracle
logp = Oracle(name = 'LogP')
qed = Oracle(name = 'QED')
jnk = Oracle(name = 'JNK3')
gsk = Oracle(name = 'GSK3B')
drd2 = Oracle(name = 'DRD2')
_7l11 = Oracle(name = '7l11_docking')
_drd3 = Oracle(name = 'drd3_docking')


def dock_drd3(smi):
    if smi is None:
        return 0.0
    else:
        try:
            return - _drd3(smi)
        except:
            return 0.0

def dock_7l11(smi):
    if smi is None:
        return 0.0
    else:
        try:
            return - _7l11(smi)
        except:
            return 0.0


def fitness(embs, _pool, obj):
    results = _pool.map(decode.func, embs)
    smiles = [r[0] for r in results]
    trees = [r[1] for r in results]
    if obj == 'qed':
        scores = [qed(smi) for smi in smiles]
    elif obj == 'logp':
        scores = [logp(smi) for smi in smiles]
    elif obj == 'jnk':
        scores = [jnk(smi) if smi is not None else 0.0 for smi in smiles]
    elif obj == 'gsk':
        scores = [gsk(smi) if smi is not None else 0.0 for smi in smiles]
    elif obj == 'drd2':
        scores = [drd2(smi) if smi is not None else 0.0 for smi in smiles]
    elif obj == '7l11':
        scores = [dock_7l11(smi) for smi in smiles]
    elif obj == 'drd3':
        scores = [dock_drd3(smi) for smi in smiles]
    else:
        raise ValueError('Objective function not implemneted')
    return scores, smiles, trees


def distribution_schedule(n, total):
    if n < 4 * total/5:
        return 'linear'
    else:
        return 'softmax_linear'

def num_mut_per_ele_scheduler(n, total):
    return 24
#     if n < total/2:
#         return 256
#     else:
#         return 512

def mut_probability_scheduler(n, total):
    if n < total/2:
        return 0.5
    else:
        return 0.5

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", type=str, default=None,
                                        help="A file contains the starting mating pool.")
    parser.add_argument("--objective", type=str, default="qed",
                                        help="Objective function to optimize")
    parser.add_argument("--radius", type=int, default=2,
                                help="Radius for Morgan fingerprint.")
    parser.add_argument("--nbits", type=int, default=4096,
                                    help="Number of Bits for Morgan fingerprint.")
    parser.add_argument("--num_population", type=int, default=100,
                                    help="Number of parents sets to keep.")
    parser.add_argument("--num_offspring", type=int, default=300,
                                    help="Number of offsprings to generate each iteration.")
    parser.add_argument("--num_gen", type=int, default=30,
                                    help="Number of generations to proceed.")
    parser.add_argument("--ncpu", type=int, default=16,
                                    help="Number of cpus")
    parser.add_argument("--mut_probability", type=float, default=0.5,
                                        help="Probability to mutate for one offspring.")
    parser.add_argument("--num_mut_per_ele", type=int, default=1,
                                        help="Number of bits to mutate in one fingerprint.")
    parser.add_argument('--restart', action='store_true')                                
    parser.add_argument("--seed", type=int, default=1,
                                        help="Random seed.")
    args = parser.parse_args()

    np.random.seed(args.seed)

    if args.restart:
        population = np.load(args.input_file)
        print(f"Starting with {len(population)} fps from {args.input_file}")
    else:
        if args.input_file is None:
            population = np.ceil(np.random.random(size=(args.num_population, args.nbits)) * 2 - 1)
            print(f"Starting with {args.num_population} fps with {args.nbits} bits")
        else:
            starting_smiles = pd.read_csv(args.input_file).sample(args.num_population)
            starting_smiles = starting_smiles['smiles'].tolist()
            population = np.array([decode.mol_fp(smi, args.radius, args.nbits) for smi in starting_smiles])
            population = population.reshape((population.shape[0], population.shape[2]))
            print(f"Starting with {len(starting_smiles)} fps from {args.input_file}")

    with mp.Pool(processes=args.ncpu) as pool:
        scores, mols, trees = fitness(population, pool, args.objective)
    scores = np.array(scores)
    score_x = np.argsort(scores)
    population = population[score_x[::-1]]
    mols = [mols[i] for i in score_x[::-1]]
    scores = scores[score_x[::-1]]
    print(f"Initial: {scores.mean():.3f} +/- {scores.std():.3f}")
    print(f"Scores: {scores}")
    print(f"Top-3 Smiles: {mols[:3]}")
    print()

    recent_scores = []

    for n in range(args.num_gen):

        t = time.time()

        dist_ = distribution_schedule(n, args.num_gen)
        num_mut_per_ele_ = num_mut_per_ele_scheduler(n, args.num_gen)
        mut_probability_ = mut_probability_scheduler(n, args.num_gen)

        offspring = crossover(population, args.num_offspring, distribution=dist_)
        offspring = mutation(offspring, num_mut_per_ele=num_mut_per_ele_, mut_probability=mut_probability_)
        new_population = np.unique(np.concatenate([population, offspring], axis=0), axis=0)
        with mp.Pool(processes=args.ncpu) as pool:
            new_scores, new_mols, trees = fitness(new_population, pool, args.objective)
        new_scores = np.array(new_scores)
        # import ipdb; ipdb.set_trace()
        scores = []
        mols = []

        parent_idx = 0
        indices_to_print = []
        while parent_idx < args.num_population:
            max_score_idx = np.where(new_scores == np.max(new_scores))[0][0]
            if new_mols[max_score_idx] not in mols:
                indices_to_print.append(max_score_idx)
                scores.append(new_scores[max_score_idx])
                mols.append(new_mols[max_score_idx])
                population[parent_idx, :] = new_population[max_score_idx, :]
                new_scores[max_score_idx] = -999999
                parent_idx += 1
            else:
                new_scores[max_score_idx] = -999999

        scores = np.array(scores)
        print(f"Generation {n+1}: {scores.mean():.3f} +/- {scores.std():.3f}")
        print(f"Scores: {scores}")
        print(f"Top-3 Smiles: {mols[:3]}")
        print(f"Consumed time: {(time.time() - t):.3f} s")
        print()
        for i in range(3):
            trees[indices_to_print[i]]._print()
        print()

        recent_scores.append(scores.mean())
        if len(recent_scores) > 10:
            del recent_scores[0]

        np.save('population_' + args.objective + '_' + str(n+1) + '.npy', population)

        data = {'objective': args.objective, 
                'top1': np.mean(scores[:1]),
                'top10': np.mean(scores[:10]),
                'top100': np.mean(scores[:100]),
                'smiles': mols, 
                'scores': scores.tolist()}
        with open('opt_' + args.objective + '.json', 'w') as f:
            json.dump(data, f)

        if n > 30 and recent_scores[-1] - recent_scores[0] < 0.01:
            print("Early Stop!")
            break

    data = {'objective': args.objective, 
            'top1': np.mean(scores[:1]),
            'top10': np.mean(scores[:10]),
            'top100': np.mean(scores[:100]),
            'smiles': mols, 
            'scores': scores.tolist()}
    with open('opt_' + args.objective + '.json', 'w') as f:
        json.dump(data, f)

    np.save('population_' + args.objective + '.npy', population)

