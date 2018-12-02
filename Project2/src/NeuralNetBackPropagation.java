import opt.OptimizationAlgorithm;
import shared.*;
import func.nn.backprop.BackPropagationNetwork;
import func.nn.backprop.BackPropagationNetworkFactory;
import func.nn.backprop.BatchBackPropagationTrainer;
import func.nn.backprop.RPROPUpdateRule;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.text.DecimalFormat;
import java.util.Scanner;

/**
 * A simple classification test
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class NeuralNetBackPropagation {
    private static Instance[] test_instances = new Instance[9041];
    private static Instance[] instances = initializeInstances(test_instances);
    private static int inputLayer = 16, hiddenLayer = 5, outputLayer = 1, trainingIterations = 1000;
    private static ErrorMeasure measure = new SumOfSquaresError();
    private static DecimalFormat df = new DecimalFormat("0.000");


    /**
     * Tests out the perceptron with the classic xor test
     *
     * @param args ignored
     */
    public static void main(String[] args) {
        BackPropagationNetworkFactory factory =
                new BackPropagationNetworkFactory();

        BackPropagationNetwork network = factory.createClassificationNetwork(
                new int[]{inputLayer, hiddenLayer, outputLayer});
        DataSet set = new DataSet(instances);
        ConvergenceTrainer trainer = new ConvergenceTrainer(
                new BatchBackPropagationTrainer(set, network,
                        new SumOfSquaresError(), new RPROPUpdateRule()), 0.05, 1000);

        double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0, correct_train = 0, incorrect_train = 0;

        trainer.train();

        end = System.nanoTime();
        trainingTime = end - start;
        trainingTime /= Math.pow(10,9);

        double predicted_train, actual_train;
        start = System.nanoTime();
        for(int j = 0; j < 36169; j++) {
            network.setInputValues(instances[j].getData());
            network.run();

            predicted_train = Double.parseDouble(instances[j].getLabel().toString());
            actual_train = Double.parseDouble(network.getOutputValues().toString());

            double trash = Math.abs(predicted_train - actual_train) < 0.5 ? correct_train++ : incorrect_train++;
        }
        end = System.nanoTime();
        testingTime = end - start;
        testingTime /= Math.pow(10,9);
        String results =  "\nResults for back propagation training" + ": \nCorrectly classified " + correct_train + " instances." +
                "\nIncorrectly classified " + incorrect_train + " instances.\nPercent correctly classified: "
                + df.format(correct_train/(correct_train+incorrect_train)*100) + "%\nTraining time: " + df.format(trainingTime)
                + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";
        System.out.println(results);
        double predicted, actual;

        start = System.nanoTime();
        for(int j = 0; j < 9041; j++) {
            network.setInputValues(test_instances[j].getData());
            network.run();

            predicted = Double.parseDouble(test_instances[j].getLabel().toString());
            actual = Double.parseDouble(network.getOutputValues().toString());

            double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;
        }
        end = System.nanoTime();
        testingTime = end - start;
        testingTime /= Math.pow(10,9);
        String results_test =  "\nResults for back propagation testing" + ": \nCorrectly classified " + correct + " instances." +
                "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                + df.format(correct/(correct+incorrect)*100) + "%\nTraining time: " + df.format(trainingTime)
                + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";
        System.out.println(results_test);
    }

    private static Instance[] initializeInstances(Instance[] test_instances) {

        double[][][] attributes = new double[36169][][];
        double[][][] test_attributes = new double[9041][][];

        try {
            BufferedReader br = new BufferedReader(new FileReader(new File("src/bank-data.txt")));

            for (int i = 0; i < attributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                attributes[i] = new double[2][];
                attributes[i][0] = new double[16]; // 16 attributes
                attributes[i][1] = new double[1];

                for (int j = 0; j < 7; j++)
                    attributes[i][0][j] = Double.parseDouble(scan.next());

                attributes[i][1][0] = Double.parseDouble(scan.next());
            }
            for (int i = 0; i < 9041; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                test_attributes[i] = new double[2][];
                test_attributes[i][0] = new double[16]; // 16 attributes
                test_attributes[i][1] = new double[1];

                for (int j = 0; j < 7; j++)
                    test_attributes[i][0][j] = Double.parseDouble(scan.next());

                test_attributes[i][1][0] = Double.parseDouble(scan.next());
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        Instance[] instances = new Instance[attributes.length];

        for (int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);
            instances[i].setLabel(new Instance(attributes[i][1][0]));
        }
        for (int i = 0; i < test_instances.length; i++) {
            test_instances[i] = new Instance(attributes[i][0]);
            test_instances[i].setLabel(new Instance(attributes[i][1][0]));
        }
        return instances;

    }

}

