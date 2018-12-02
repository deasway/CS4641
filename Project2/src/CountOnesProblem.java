import java.util.Arrays;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.ga.UniformCrossOver;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;
public class CountOnesProblem {

    public static void main(String[] args) {
        int N = 200;
        int a[] = new int[N];

        Arrays.fill(a, 2);

        EvaluationFunction evaluationFunction = new CountOnesEvaluationFunction();
        Distribution distribution = new DiscreteUniformDistribution(a);
        NeighborFunction neighborFunction = new DiscreteChangeOneNeighbor(a);
        MutationFunction mutationFunction = new DiscreteChangeOneMutation(a);
        CrossoverFunction crossoverFunction = new UniformCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, a);
        HillClimbingProblem hcp = new GenericHillClimbingProblem(evaluationFunction, distribution, neighborFunction);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(evaluationFunction, distribution, mutationFunction, crossoverFunction);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(evaluationFunction, distribution, df);

        double start = System.nanoTime(), end, time;
        FixedIterationTrainer fit = null;

        double times[] = new double[30];

        // Randomized Hill Climbing
        System.out.println("RHC--------------");
        for (int i = 100; i <= 3000; i+= 100) {
            RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);

            fit= new FixedIterationTrainer(rhc, i);
            start = System.nanoTime();
            fit.train();
            end = System.nanoTime();
            time = end - start;
            time /= Math.pow(10,9);
//            System.out.println(evaluationFunction.value(rhc.getOptimal()));

        }
        // Simulated Annealing
        System.out.println("SA--------------");

        for (int i = 100; i <= 3000; i+= 100) {
            SimulatedAnnealing sa = new SimulatedAnnealing(1E11, .95, hcp);

            fit = new FixedIterationTrainer(sa, i);
            start = System.nanoTime();
            fit.train();
            end = System.nanoTime();
            time = end - start;
            time /= Math.pow(10,9);
//            System.out.println(evaluationFunction.value(sa.getOptimal()));
            times[i / 100 - 1] = time;

        }
        // Genetic Algorithm
        System.out.println("GA--------------");
        for (int i = 100; i <= 3000; i+= 100) {
            StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 100, 10, gap);
            fit = new FixedIterationTrainer(ga, i);
            start = System.nanoTime();
            fit.train();
            end = System.nanoTime();
            time = end - start;
            time /= Math.pow(10,9);
//            System.out.println(evaluationFunction.value(ga.getOptimal()));
            System.out.println(time / times[i / 100 - 1]);
        }

        System.out.println("MIMIC------------");

        for (int i = 100; i <= 3000; i+= 100) {
            MIMIC mimic = new MIMIC(200, 20, pop);
            fit = new FixedIterationTrainer(mimic, i);
            start = System.nanoTime();
            fit.train();
            end = System.nanoTime();
            time = end - start;
            time /= Math.pow(10,9);
            System.out.println(evaluationFunction.value(mimic.getOptimal()));
            System.out.println(time / times[i / 100 - 1]);
        }
    }

}
