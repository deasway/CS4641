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
import opt.example.FourPeaksEvaluationFunction;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.SingleCrossOver;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;
import shared.Instance;

import java.util.Arrays;


public class FourPeaksProblem {

    /*Create a four peaks problem with N = 200, T = N / 5 problem is defined at
    https://pdfs.semanticscholar.org/cd4f/e89d8dd6060e2957041f90fc699a30058d01.pdf
     */
    public static void main(String[] args) {
        int N = 100;
        int T = N / 10;
        int[] a = new int[N];
        Arrays.fill(a, 2);


        EvaluationFunction evaluationFunction = new FourPeaksEvaluationFunction(T);
        Distribution distribution = new DiscreteUniformDistribution(a);
        NeighborFunction neighborFunction = new DiscreteChangeOneNeighbor(a);
        MutationFunction mutationFunction = new DiscreteChangeOneMutation(a);
        CrossoverFunction crossoverFunction = new SingleCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, a);

        ProbabilisticOptimizationProblem pop = new
                GenericProbabilisticOptimizationProblem(evaluationFunction,
                distribution, df);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(
                evaluationFunction, distribution, mutationFunction,
                crossoverFunction);
        HillClimbingProblem hcp = new GenericHillClimbingProblem(
                evaluationFunction, distribution, neighborFunction);

        double start = System.nanoTime(), end, time;
        FixedIterationTrainer fit = null;

        // Randomized Hill Climbing
        System.out.println("RHC--------------");
        for (int i = 500; i <= 15000; i += 500) {
            RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);

            fit= new FixedIterationTrainer(rhc, i);
            start = System.nanoTime();
            fit.train();
            end = System.nanoTime();
            time = end - start;
            time /= Math.pow(10,9);
            System.out.println(evaluationFunction.value(rhc.getOptimal()));
            System.out.println(time);

        }
        // Simulated Annealing
        System.out.println("SA--------------");

        for (int i = 500; i <= 15000; i+= 500) {
            SimulatedAnnealing sa = new SimulatedAnnealing(1E11, .95, hcp);

            fit = new FixedIterationTrainer(sa, i);
            start = System.nanoTime();
            fit.train();
            end = System.nanoTime();
            time = end - start;
            time /= Math.pow(10,9);
            System.out.println(evaluationFunction.value(sa.getOptimal()));
//            System.out.println(time);

        }
        // Genetic Algorithm
        System.out.println("GA--------------");
        double times[] = new double[30];
        for (int i = 500; i <= 15000; i+= 500) {
            StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 100, 10, gap);
            fit = new FixedIterationTrainer(ga, i);
            start = System.nanoTime();
            fit.train();
            end = System.nanoTime();
            time = end - start;
            time /= Math.pow(10,9);
//            System.out.println(evaluationFunction.value(ga.getOptimal()));
            times[i / 500 - 1] = time;
            System.out.println(time);

        }

        System.out.println("MIMIC------------");

        for (int i = 500; i <= 15000; i+= 500) {
            MIMIC mimic = new MIMIC(200, 20, pop);
            fit = new FixedIterationTrainer(mimic, i);
            start = System.nanoTime();
            fit.train();
            end = System.nanoTime();
            time = end - start;
            time /= Math.pow(10,9);
//            System.out.println(evaluationFunction.value(mimic.getOptimal()));
            System.out.println(time / times[i / 500 - 1]);

        }

    }



}
