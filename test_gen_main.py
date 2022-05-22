import numpy as np

from myLib.Result import SingleAttackResult
from myLib.img_mutations import get_img_mutations
from myLib.covered_states import CoveredStates
from proj_utils import summary_attack_results
import time


# attack mode:
# help="a for normal, b for no probability guide, c for no state function")


def tf_gen(args, inputs_set, logger, save_dir, predict_f, preprocessing):
    number_of_data = inputs_set.len
    success_iter = np.ones([number_of_data]) * -1
    l2distances = np.ones([number_of_data]) * -1
    newl2distances = np.ones([number_of_data]) * -1
    num_reductions = []

    # -1 for failure
    # 0 for no need to search
    # > 0 means the iteration where the objective is reached

    for idx in range(0, number_of_data):

        seed_file = inputs_set[idx]
        raw_seed_input = seed_file["img"]
        seed_label = seed_file["label"]

        logger("Img idx {} label {}".format(idx, seed_label))

        # if len(raw_seed_input.shape) == 2:
        #     raw_seed_input = raw_seed_input[np.newaxis, ...]

        if args.attack_mode == "a" or args.attack_mode == "b" or args.attack_mode == "d":
            covered_states = CoveredStates()
        elif args.attack_mode == "c":
            pass
        else:
            raise NotImplementedError

        # record the seen state of two models

        mutation = get_img_mutations()

        if args.attack_mode == "a" or args.attack_mode == "c" or args.attack_mode == "d":
            from myLib.probability_img_mutations import ProbabilityImgMutations as ImgMutations
        elif args.attack_mode == "b":
            from myLib.probability_img_mutations import RandomImgMutations as ImgMutations
        else:
            raise NotImplementedError

        p_mutation = ImgMutations(mutation, args.seed)

        # record the probability of each mutation method
        # it will also somehow decide which mutation will be applied next

        seed_org_result, seed_cps_result = predict_f(preprocessing(raw_seed_input))

        result = SingleAttackResult(raw_seed_input, seed_label, idx, seed_org_result, seed_cps_result, save_dir)
        start_time = time.time()

        if seed_org_result.label != seed_cps_result.label:
            logger("No need to search: org: {} vs cps: {}".format(seed_org_result.label, seed_cps_result.label))
            success_iter[idx] = 0
            result.update_results(None, seed_org_result,
                                  seed_cps_result, 0)

        else:
            if args.attack_mode == "a" or args.attack_mode == "b" or args.attack_mode == "d":
                _, _ = covered_states.update_function(np.hstack([seed_org_result.vec, seed_cps_result.vec]))
                from myLib.fitnessValue import StateFitnessValue as FitnessValue
                best_fitness_value = FitnessValue(False, 0)
            elif args.attack_mode == "c":
                from myLib.fitnessValue import DiffProbFitnessValue as FitnessValue
                best_fitness_value = FitnessValue(-1)
            else:
                raise NotImplementedError

            latest_img = np.copy(raw_seed_input)
            # a.k.a last unsuccessful failed mutant

            last_mutation_operator = None

            for iteration in range(1, args.maxit + 1):
                logger("Iteration： {}".format(iteration))
                if time.time() - start_time > args.timeout:
                    logger("Time Out")
                    break

                m = p_mutation.choose_mutator(last_mutation_operator)
                m.total += 1
                logger("Mutator :{}".format(m.name))
                new_img = m.mut(np.copy(latest_img))

                org_result, cps_result = predict_f(
                    preprocessing(new_img))

                if org_result.label != cps_result.label:
                    logger("Found: org: {} vs cps: {}".format(org_result.label, cps_result.label))
                    success_iter[idx] = iteration
                    m.delta_bigger_than_zero += 1
                    result.update_results(new_img, org_result, cps_result, iteration)
                    l2dist = np.linalg.norm((new_img - raw_seed_input).flatten(), ord=2)
                    l2distances[idx] = l2dist
                    break
                else:

                    diff_prob = org_result.prob[0] - cps_result.prob[0]
                    # l2 = np.linalg.norm((latest_img - new_img).flatten(), ord=2)
                    # linf = np.linalg.norm((latest_img - new_img).flatten(), ord=np.inf)
                    # print(f"{m.name}, {l2:.3f}, {linf}")

                    if args.attack_mode == "a" or args.attack_mode == "b" or args.attack_mode == "d":
                        # check if new path is triggered:
                        # fitness_value = 0
                        # m.update_p(0.0)
                        coverage = np.hstack([org_result.vec, cps_result.vec])
                        # perhaps we need to change the coverage

                        add_to_corpus, distance = covered_states.update_function(coverage)
                        # print(f"add to corpus: {add_to_corpus}")
                        fitness_value = FitnessValue(add_to_corpus, diff_prob)
                    else:
                        # mode c: only consider the probability difference
                        fitness_value = FitnessValue(diff_prob)

                    # TODO: need to rethink the definition of the return value.
                    # if abs(org_probs - cps_probs) - abs(org_probs - cps_probs):
                    #     pass
                    # TODO: we can check the probs as well....

                    if fitness_value.better_than(best_fitness_value):
                        update_str = "update fitness value from {}".format(best_fitness_value)
                        best_fitness_value = fitness_value
                        if args.attack_mode == "d":
                            l2dist = np.linalg.norm((latest_img - new_img).flatten(), ord=2)
                            m.delta_bigger_than_zero += 1 / (l2dist ** -2)

                        else:
                            # m.delta_bigger_than_zero += 0.5
                            m.delta_bigger_than_zero += 1
                        latest_img = np.copy(new_img)
                        last_mutation_operator = m
                        update_str += " to {}".format(best_fitness_value)
                        logger(update_str)

                    logger("Best " + str(best_fitness_value))

        result.save()

    # compute the results
    summary_attack_results(success_iter, logger, args.attack_mode)