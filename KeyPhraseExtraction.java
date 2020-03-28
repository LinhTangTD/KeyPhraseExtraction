/**
 * @author Linh Tang & Yolanda Jiang
 * @date March.27
 */
package assignment3;

import java.io.*;
import java.nio.file.*;
import java.util.*;

import edu.stanford.nlp.tagger.maxent.MaxentTagger;

public class KeyPhraseExtraction {

	/**
	 * Check Stop Words
	 * 
	 * @param a string of n words (n depends on which ngrams are being generated)
	 * @return a boolean indicate if given string is stop word(s)
	 * @throws IOException
	 */
	public static boolean checkStopwords(String s) throws IOException {
		// Load in a stop-list from the given stopwords file
		List<String> stopwords = Files.readAllLines(Paths.get("stopwords.txt"));
		String[] temp = s.split(" ");

		// All words in the string beginning or ending in the stop-list will count the
		// whole string as stop word(s)
		return stopwords.contains(temp[0].toLowerCase()) || stopwords.contains(temp[temp.length - 1].toLowerCase());
	}

	/**
	 * Generate a list of N-grams
	 * 
	 * @param n
	 * @param filename
	 * @return a list of string (each string is n-word)
	 * @throws IOException
	 */
	public static List<String> genNgram(int n, String filename) throws IOException {

		// Load in the file into one string
		String original = new String(Files.readAllBytes(Paths.get(filename)));

		if (n == 1) // if we need to generate an unigram
			return Arrays.asList(original.replaceAll("[\\p{Punct}&&[^_-]]+", "").split("\\s+"));

		// reformat input
		original = original.replaceAll("\n\t", " ");
		original = original.replaceAll("\n", "\\.");

		// split into sentences
		String[] sentences = original.split("[\\p{Punct}&&[^_-]]+");

		// Generate n-grams
		List<String> ngram = new ArrayList<String>();
		for (String sen : sentences) {

			// Remove punctuation and spit original file into separate word by space
			String[] words = sen.split("\\s+");

			// Generate n-gram and check for stop word
			int len = words.length;
			for (int i = 0; i < len - n; i++) {

				// build string of n words
				StringBuilder word = new StringBuilder();
				for (int j = i; j < i + n - 1; j++)
					word.append(words[j] + " ");
				word.append(words[i + n - 1]);

				// Check if it's a stop word(s) and add to return list
				if (!checkStopwords(word.toString()))
					ngram.add(word.toString());
			}
		}
		return ngram;
	}

	/**
	 * Get a list of files by specific extension from a folder (directory)
	 * 
	 * @param folder
	 * @param ext
	 * @return list of files with specific extension
	 * @throws IOException
	 */
	public static List<File> filterFile(String foldername, String extension) throws IOException {

		// Open the folder
		File dir = new File(foldername);

		// Get a list of files by extension from a folder
		List<File> files = new ArrayList<File>();
		for (File file : dir.listFiles()) {
			if (file.getName().endsWith(extension))
				files.add(file);
		}
		return files;
	}

	/**
	 * Calculate score for each term in each document
	 * 
	 * @param term
	 * @param doc
	 * @param collection
	 * @return double - the score before POS-curving
	 */
	public static double calculator(String term, List<String> doc, List<List<String>> collection) {

		// Calculate TermFrequency (TF)
		double freq_tf = 0.0;
		for (String word : doc)
			if (term.equals(word))
				freq_tf++;
		double tf = Math.log(1 + freq_tf);

		// Calculate InverseDocumentFrequency (IDF)
		double freg_idf = 0.0;
		for (List<String> document : collection)
			if (document.contains(term))
				freg_idf++;
		double idf = Math.log(collection.size() / freg_idf);

		// Calculate Relative position of the first-occurrence (relpos)
		String doc_joined = String.join(" ", doc);
		int idx = doc_joined.indexOf(term);
		double relpos = (double) idx / doc_joined.length();

		return tf * idf * relpos;
	}

	/**
	 * Find the top 5 highest scored keywords
	 * 
	 * @param hm, a dictionary of scores of all terms in a document
	 * @return a list of top5 highest keywords
	 */
	public static List<String> top5(HashMap<String, Double> hm) {

		// Sort the list of keywords by their scores
		List<Map.Entry<String, Double>> l = new LinkedList<Map.Entry<String, Double>>(hm.entrySet());
		Collections.sort(l, new Comparator<Map.Entry<String, Double>>() {
			public int compare(Map.Entry<String, Double> o1, Map.Entry<String, Double> o2) {
				return (o1.getValue()).compareTo(o2.getValue());
			}
		});
		List<String> top5 = new ArrayList<String>();

		// Set the number of keywords to return
		int max = 5;

		// If the original document has fewer than 5 words
		if (hm.size() < 5)
			max = hm.size();
		for (int i = 0; i < max; i++) {
			top5.add(l.get(i).getKey());
		}

		return top5;
	}

	/**
	 * Generate a list of 3 ngrams: unigram, bigram, trigram
	 * 
	 * @param filename
	 * @return a list of string
	 * @throws IOException
	 */
	public static List<String> allNgrams(int n, String filename) throws IOException {
		List<String> result = new ArrayList<String>();
		for (int i = 1; i <= n; i++) {
			List<String> ngram = genNgram(i, filename);
			result.addAll(ngram);
		}
		return result;
	}

	/**
	 * Check if the given phrase is important words using the Stanford POS-tagger
	 * model
	 * 
	 * @param phrase
	 * @return a boolean
	 * @throws IOException
	 * @throws ClassNotFoundException
	 */
	public static boolean checkTag(String phrase) throws IOException, ClassNotFoundException {
		Properties cfg;
		cfg = new Properties();
		cfg.setProperty("outputFormatOptions", "lemmatize");
		cfg.setProperty("outputFormat", "slashTags");

		MaxentTagger tagger = new MaxentTagger("stanford-postagger-2018-10-16/models/english-left3words-distsim.tagger",
				cfg);
		String[] tagged = tagger.tagString(phrase).split(" ");

		HashMap<String, String> WordTagDict = new HashMap<String, String>();
		for (String s : tagged) {
			String word = s.split("_")[0];
			String tag = s.split("_")[1];
			WordTagDict.put(word, tag);
		}

		for (String tag : WordTagDict.values()) {
			if (!tag.startsWith("N") & !tag.startsWith("J"))
				return false;
		}
		return true;
	}

	/**
	 * Model calculate precision scores for the folder
	 * 
	 * @param n
	 * @param model
	 * @return a list of scores(average score, highest score, lowest score)
	 * @throws IOException
	 * @throws ClassNotFoundException
	 */
	public static List<Double> Model(int n, String model) throws IOException, ClassNotFoundException {

		List<File> files = filterFile("Training", ".abstr");
		Collections.sort(files);

		List<List<String>> collection = new ArrayList<List<String>>();
		for (File file : files) {
			String filename = file.getAbsolutePath();
			List<String> doc = new ArrayList<String>();

			if (model.equals("ngram")) {
				doc = genNgram(n, filename);
			} else if (model.equals("POStagger")) {
				doc = allNgrams(n, filename);
			}
			collection.add(doc);
		}

		List<File> test_files = filterFile("Training", ".uncontr");
		Collections.sort(test_files);

		List<List<String>> test_collection = new ArrayList<List<String>>();
		for (File file : test_files) {
			String filename = file.getAbsolutePath();
			String original = new String(Files.readAllBytes(Paths.get(filename)));
			original = original.replaceAll("\\s+", " ");
			original = original.replaceAll(" $", "");
			List<String> words = Arrays.asList(original.split("; "));
			test_collection.add(words);
		}

		int len = collection.size();
		double total_precision = 0;
		double best_precision = 0;
		double worst_precision = 1;

		for (int i = 0; i < len; i++) {
			List<String> doc = collection.get(i);
			HashMap<String, Double> termRank = new HashMap<String, Double>();
			for (String term : doc) {
				double score = calculator(term, doc, collection);
				if (model.equals("POStag") && checkTag(term)) {
					score *= 1.66;
				}
				termRank.put(term, score);
			}

			List<String> KeyWords = top5(termRank);

			double m = 0;
			List<String> test_KeyWords = test_collection.get(i);
			for (String kw : KeyWords)
				if (test_KeyWords.contains(kw.toLowerCase()))
					m++;

			double precision = m / KeyWords.size();
			if (precision > best_precision) {
				best_precision = precision;
			} else if (precision < worst_precision) {
				worst_precision = precision;
			}
			total_precision += precision;
		}
		double ave_precision = total_precision / len;


		List<Double> result = new ArrayList<Double>();
		result.add(ave_precision);
		result.add(best_precision);
		result.add(worst_precision);

		return result;
	}

	/**
	 * wordDegree calculate the degree of word 
	 * 
	 * @param s
	 * @param words
	 * @param content
	 * @return degree, double
	 */
	public static Double wordDegree(String s, String[] words, List<String> content) {
		HashMap<String, Integer> hm = new HashMap<String, Integer>();
		int counter = 0;
		for (int i = 0; i < words.length-1; i++) {
			if (words[i].equals(s)) {
				if (content.contains(words[i + 1])) {
					if (hm.containsKey(words[i + 1])) {
						int newV = hm.get(words[i + 1]) + 1;
						hm.put(words[i + 1], newV);
					} else {
						hm.put(words[i + 1], 1);
					}
				}
				counter++;
			}
		}
		hm.put(s, counter);
		int sum = 0;
		for (int i : hm.values()) {
			sum += i;
		}
		Double score = (double) sum / (double) hm.get(s);
		return score;
	}

	/**
	 * getContentPhrases generate a list of phrases that only consist of content string 
	 * 
	 * @param phrase
	 * @param content
	 * @return list of string
	 */
	public static List<String> genContentPhrases(String[] phrase, List<String> content) {
		List<String> contentPhrases = new ArrayList<String>();
	 
		for (String s : phrase) {
			String[] words = s.split(" ");
			StringBuilder sb = new StringBuilder();
			if (words.length != 1) {
			for (String w : words) {
				
				if (content.contains(w)) {
					sb.append(w + " ");
				} else if(sb.length() != 0){
					contentPhrases.add(sb.toString());
					sb.setLength(0);
				}
			}
			contentPhrases.add(sb.toString());
			sb.setLength(0);
			} else {
				if(content.contains(s)) {
					contentPhrases.add(s);
				}
			}
			
		}
		return contentPhrases;
	}

	/**
	 * rake calculate the degree of each content phrases
	 * 
	 * @param input
	 * @return hashmap contains content phrases as key and degree as values
	 * @throws IOException
	 */
	public static HashMap<String, Double> rake(String input) throws IOException {
		List<String> stopwords = Files.readAllLines(Paths.get("stopwords.txt"));
		String original = new String(Files.readAllBytes(Paths.get(input)));

		original = original.replaceAll("\n\t", " ");
		original = original.replaceAll("\n|\r", "\\.");

		String[] phrases = original.split("[\\p{Punct}&&[^_-]]+");

		original = original.replaceAll("[\\p{Punct}&&[^_-]]+", " ");
		String[] words = original.split("\\s+");
		List<String> content = new ArrayList<String>();
		for (String s : words) {
			if (!stopwords.contains(s)) {
				content.add(s);
			}
		}
		
		HashMap<String, Double> wordHm = new HashMap<String, Double>();
		for (String s : content) {
			wordHm.put(s, wordDegree(s, words, content));
		}
		
		List<String> contentPhrases = genContentPhrases(phrases, content);
		HashMap<String, Double> phraseHm = new HashMap<String, Double>();
		for (String s : contentPhrases) {
			Double score = 0.0;
			if (s.length() == 1) {
				phraseHm.put(s, wordHm.get(s));
			} else {
				String[] l = s.split(" ");
				for (String st : l) {
					if(!st.equals("")) {
					score += wordHm.get(st);
					} 
				}
			}
			phraseHm.put(s, score);
		}

		return phraseHm;

	}

	/**
	 * rakeModel calculate precision score for the folder
	 * @return a list of scores, average score, best score, worst score
	 * @throws IOException
	 */
	public static List<Double> rakeModel() throws IOException {
		List<File> files = filterFile("Training", ".abstr");
		Collections.sort(files);

		List<File> test_files = filterFile("Training", ".uncontr");
		Collections.sort(test_files);

		List<List<String>> test_collection = new ArrayList<List<String>>();
		for (File file : test_files) {
			String filename = file.getAbsolutePath();
			String original = new String(Files.readAllBytes(Paths.get(filename)));
			original = original.replaceAll("\\s+", " ");
			original = original.replaceAll(" $", "");
			List<String> words = Arrays.asList(original.split("; "));
			test_collection.add(words);
		}

		int len = files.size();
		double total_precision = 0;
		double best_precision = 0;
		double worst_precision = 1;
		
		for (int i = 0; i < len; i++) {
			String filename = files.get(i).getAbsolutePath();
			List<String> KeyWords = top5(rake(filename));

			double m = 0;
			List<String> test_KeyWords = test_collection.get(i);
			for (String kw : KeyWords)
				if (test_KeyWords.contains(kw.toLowerCase()))
					m++;

			double precision = m / KeyWords.size();

			total_precision += precision;

			if (precision > best_precision) {
				best_precision = precision;
			} else if (precision < worst_precision) {
				worst_precision = precision;
			}
			total_precision += precision;
		}
		double ave_precision = total_precision / len; 
		
		List<Double> result = new ArrayList<Double>();
		result.add(ave_precision);
		result.add(best_precision);
		result.add(worst_precision);

		return result;
	}

	public static void main(String[] args) throws IOException, ClassNotFoundException {
		
		List<Double> RAKE = rakeModel();
		List<Double> unigram = Model(1, "ngram");
		List<Double> bigram = Model(2, "ngram");
		List<Double> trigram = Model(3, "ngram");
		List<Double> POSTagger = Model(3, "POStagger");
		
		PrintStream out = new PrintStream(new FileOutputStream("report.txt"));
		System.setOut(out);
		System.out.format("%13s%21s%21s%21s%n", "Model", "Average Precision", "Best Precision", "Worst Precision");
		System.out.format("%13s%21f%21f%21f%n", "Unigram", unigram.get(0), unigram.get(1), unigram.get(2));
		System.out.format("%13s%21f%21f%21f%n", "Bigram", bigram.get(0), bigram.get(1), bigram.get(2));
		System.out.format("%13s%21f%21f%21f%n", "Trigram", trigram.get(0), trigram.get(1), trigram.get(2));
		System.out.format("%13s%21f%21f%21f%n", "POSTagger", POSTagger.get(0), POSTagger.get(1), POSTagger.get(2));
		System.out.format("%13s%21f%21f%21f%n", "RAKE", RAKE.get(0), RAKE.get(1), RAKE.get(2));
	
	}

}
