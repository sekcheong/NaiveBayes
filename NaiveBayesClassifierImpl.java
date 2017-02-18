/**
 * CS540 Section 1
 * Sek Cheong 
 * sek@cs.wisc.edu
 * 
 * Your implementation of a naive bayes classifier. Please implement all four
 * methods.
 */
import java.util.*;


public class NaiveBayesClassifierImpl implements NaiveBayesClassifier
{

    private int _nclasses = Label.values().length;

    int _count[] = new int[_nclasses];
    double _prob[] = new double[_nclasses];
    int _words = 0;

    private class Word implements Comparable<Word>
    {

        public Word() {
            count = new int[_nclasses];
            prob = new double[_nclasses];
            for (int i = 0; i < _nclasses; i++) {
                count[i] = 0;
                prob[i] = 0;
            }
        }

        public String value;
        public int count[];
        public double prob[];
        public double Informativeness;
        public int compareTo(Word arg0) {
            if (this.Informativeness>arg0.Informativeness) return -1;
            if (this.Informativeness<arg0.Informativeness) return 1;
            return 0;
        }
    }

    private double DELTA = 0.00001;

    private Map<String, Word> _dict = new HashMap<String, Word>();

    /**
     * Trains the classifier with the provided training data and vocabulary size
     */
    @Override
    public void train(Instance[] trainingData, int v) {

        for (Label l : Label.values()) {
            _count[l.ordinal()] = 0;
            _prob[l.ordinal()] = 0;
        }

        for (Instance ins : trainingData) {

            int cls = ins.label.ordinal();

            for (String word : ins.words) {
                Word w;

                if (!_dict.containsKey(word)) {
                    w = new Word();
                    _dict.put(word, w);
                }
                else {
                    w = _dict.get(word);
                }

                w.value = word;
                w.count[cls]++;
                _count[cls]++;
                _words++;
            }
        }

        for (Label lbl : Label.values()) {
            int l = lbl.ordinal();
            for (Word w : _dict.values()) {
                // computes p(w|l)
                w.prob[l] = (w.count[l] + DELTA) / (_count[l] + _words * DELTA);
            }
        }

        for (Label l : Label.values()) {
            _prob[l.ordinal()] = ((double) _count[l.ordinal()]) / ((double) _words);
        }

        // for (Word w: _dict.values()) {
        // System.out.printf("(%s, %d, %d, %1.10f, %1.10f)\n", w.value,
        // w.count[0], w.count[1], w.prob[0], w.prob[1]);
        // }
        // System.out.printf("P(S):%f, P(H):%f\n", _prob[0], _prob[1]);

    }

    /**
     * Returns the prior probability of the label parameter, i.e. P(SPAM) or
     * P(HAM)
     */
    @Override
    public double p_l(Label label) {
        return _prob[label.ordinal()];
    }

    /**
     * Returns the smoothed conditional probability of the word given the label,
     * i.e. P(word|SPAM) or P(word|HAM)
     */
    @Override
    public double p_w_given_l(String word, Label label) {
        if (_dict.containsKey(word)) {
            Word w = _dict.get(word);
            return w.prob[label.ordinal()];
        }
        else {
            return DELTA / ((_count[label.ordinal()] + _dict.size() * DELTA));
        }
    }

    /**
     * Classifies an array of words as either SPAM or HAM.
     */
    @Override
    public Label classify(String[] words) {
        double g = 0;
        double maxg = Double.NEGATIVE_INFINITY;
        Label maxLabel = Label.values()[0];
        for (Label l : Label.values()) {
            g = Math.log(p_l(l)) + sum_p_w_given_l(words, l);
            if (g > maxg) {
                maxg = g;
                maxLabel = l;
            }
        }
        return maxLabel;
    }

    private double sum_p_w_given_l(String[] words, Label l) {
        double ret = 0;
        for (String word : words) {
            ret = ret + Math.log(p_w_given_l(word, l));
        }
        return ret;
    }
    

    /**
     * Print out 5 most informative words.
     */
    public void show_informative_5words() {
        List<Word> words = new ArrayList<Word>();
        for (Word w : _dict.values()) {
            w.Informativeness = Math.max(p_w_given_l(w.value, Label.HAM) / p_w_given_l(w.value, Label.SPAM),
                    p_w_given_l(w.value, Label.SPAM) / p_w_given_l(w.value, Label.HAM));
            words.add(w);
        }
        Collections.sort(words);
        int n = Math.min(words.size(), 5);
        if (n > 0) {
            System.out.printf("%d most informative words:\n", n);
            for (int i = 0; i < n; i++) {
                System.out.printf("%d. %s\n", i + 1, words.get(i).value);
            }
        }
    }
}