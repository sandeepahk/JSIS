package jsis;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.StringTokenizer;
import java.util.Vector;

import uk.ac.wlv.sentistrength.SentiStrength;





public class Model {
	
	//---------------------------------------------------------------
	//	Class Variables
	//---------------------------------------------------------------
	
	public static String tassignSuffix;	//suffix for topic assignment file
	public static String thetaSuffix;		//suffix for theta (topic - document distribution) file
	public static String phiSuffix;		//suffix for phi file (topic - word distribution) file
	public static String othersSuffix; 	//suffix for containing other parameters
	public static String twordsSuffix;		//suffix for file containing words-per-topics
	
	//---------------------------------------------------------------
	//	Model Parameters and Variables
	//---------------------------------------------------------------
	
	public String wordMapFile; 		//file that contain word to id map
	public String trainlogFile; 	//training log file	
	
	public String dir;
	public String dfile;
	public String modelName;
	public int modelStatus; 		//see Constants class for status of model
	public LDADataset data;			// link to a dataset
	
	public int M; //dataset size (i.e., number of docs)
	public int V; //vocabulary size
	public int I; //number of entities
	public int S; //number of sentiment
	public int Y; //number of stance
	
	
	// temp variables for sampling
	protected double [][][] p; //size I x S x Y
	
	public double alpha_i, alpha_s, alpha_y, beta_i, beta_s, beta_y; //LDA  hyperparameters
	public double [][][] alpha_isy;
	public double [][] alphaSum_is;
	public double [][] beta_iw, beta_sw, beta_yw; //size I x V, S x v
	public double [] beta_isum, beta_ssum, beta_ysum; //sum of beta values
	public double [][] lambda_i, lambda_s, lambda_y; //size I x V, S x V---for encoding prior topic information 
	public int niters; //number of Gibbs sampling iteration
	public int liter; //the iteration at which the model was saved	
	public int savestep; //saving period
	public int twords; //print out top words per each topic
	public int updateParaSteps;// prameter update steps
	
	// Estimated/Inferenced parameters
	public double [][] theta_i, theta_s; //theta: document - topic distributions, size M x I, M x S
	public double [][][][] theta_y; //size M x I x S x I
	public double [][] phi_i, phi_s, phi_y; // phi: topic-word distributions, size I x V, S x V
	
	// Temp variables while sampling
	public Vector<Integer> [] i; //issue assignments for words, size M x doc.size()
	public Vector<Integer> [] s; //sentiment assignments for words, size M x doc.size()
	public Vector<Integer> []  y; //stance assignments for groups, size M 
	protected int [][] nwi, nws, nwy; //nw[i][j]: number of instances of word/term i assigned to topic j, size V x I, V x S
	protected int [][] ndi, nds; //nd[i][j]: number of words in document i assigned to topic j, size M x I, M x S
	protected int [] ni, ns, ny; //nwsum[j]: total number of words assigned to topic j, size E, S, T
	protected int [] nd; //ndsum[i]: total number of words in document i, size M
	protected int [][][] ndis; // size M x I x S
	protected int [][][][] ndisy; // size M x I x S x Y
	
	private ArrayList<List<String>> initialIssueWord;
	private ArrayList<List<String>> initialSentimentWord;
	private ArrayList<List<String>> initialStanceWord;
	
	public Model(){
		initialIssueWord = new ArrayList<List<String>>();
		initialSentimentWord = new ArrayList<List<String>>();
		initialStanceWord = new ArrayList<List<String>>();
		setDefaultValues();	
	}
	
	/**
	 * Set default values for variables
	 */
	public void setDefaultValues(){
		wordMapFile = "wordmap.txt";
		trainlogFile = "trainlog.txt";
		tassignSuffix = ".tassign";
		thetaSuffix = ".theta";
		phiSuffix = ".phi";
		othersSuffix = ".others";
		twordsSuffix = ".twords";
		
		dir = "./";
		dfile = "trndocs.dat";
		modelName = "model-final";
		modelStatus = Constants.MODEL_STATUS_UNKNOWN;		
		
		M = 0;
		V = 0;
		I = 100;
		S = 100;
		Y = 100;
		alpha_i = 50.0 / I;
		alpha_s = 50.0 / S;
		alpha_y = 50.0 / Y;
		beta_i = 0.1;
		beta_s = 0.1;
		beta_y = 0.1;
		niters = 2000;
		liter = 0;
		updateParaSteps = 40;
		
		i = null;
		s = null;
		y = null;
		nwi = null;
		nws = null;
		nwy = null;
		ndi = null;
		nds = null;
		ni = null;
		ns = null;
		ny = null;
		nd = null;
		ndis = null;
		ndisy = null;
		
		theta_i = null;
		theta_s = null;
		theta_y = null;
		phi_i = null;
		phi_s = null;
		phi_y = null;
		
	}
	
	/**
	 * Init parameters for estimation
	 */
	public boolean initNewModel(LDACmdOption option){
		if (!init(option))
			return false;
		
		p = new double[I][S][Y];		
		
		data = LDADataset.readDataSet(dfile, dir);
		if (data == null){
			System.out.println("Fail to read training data!\n");
			return false;
		}
		
		//+ allocate memory and assign values for variables		
		M = data.M;
		V = data.V;
		
		System.out.println(data.W);
		//dir = option.dir;
		savestep = option.savestep;
		updateParaSteps = option.updateParaSteps;
		
		// K: from command line or default value
	    // alpha, beta: from command line or default values
	    // niters, savestep: from command line or default values

		nwi = new int[V][I];
		for (int w = 0; w < V; w++){
			for (int i = 0; i < I; i++){
				nwi[w][i] = 0;
			}
		}
		
		nws = new int[V][S];
		for (int w = 0; w < V; w++){
			for (int s = 0; s < S; s++){
				nws[w][s] = 0;
			}
		}
		
		nwy = new int[V][Y];
		for (int w = 0; w < V; w++){
			for (int y = 0; y < Y; y++){
				nwy[w][y] = 0;
			}
		}
		
		ni =  new int[I];
		for (int i = 0; i < I; i++) {
			ni[i] = 0;
		}
		
		ns =  new int[S];
		for (int s = 0; s < S; s++) {
			ns[s] = 0;
		}
		
		ny =  new int[Y];
		for (int y = 0; y < Y; y++) {
			ny[y] = 0;
		}
		
		ndi = new int[M][I];
		for (int m = 0; m < M; m++){
			for (int i = 0; i < I; i++) {
				ndi[m][i] = 0;
			}
		}
		

		nds = new int[M][S];
		for (int m = 0; m < M; m++){
			for (int s = 0; s < S; s++) {
				nds[m][s] = 0;
			}
		}
		
		nd = new int[M];
		for (int m = 0; m < M; m++){
			nd[m] = 0;
			
		}
		
		ndis =  new int[M][I][S];
		for (int m = 0; m < M; m++){
			for (int i = 0; i < I; i++) {
				for (int s = 0; s < S; s++) {
					ndis[m][i][s] = 0;
				}
			}
		}
		
		ndisy =  new int[M][I][S][Y];
		for (int m = 0; m < M; m++){
			for (int i = 0; i < I; i++) {
				for (int s = 0; s < S; s++) {
					for (int y = 0; y < Y; y++) {
						ndisy[m][i][s][y] = 0;
					}					
				}
			}
		}
		
		loadData();
		
		i = new Vector[M];
		s = new Vector[M];
		y = new Vector[M];
		
		SentiStrength sentiStrength = new SentiStrength(); 
		String ssthInitialisation[] = {"sentidata", "topicList/sentiment/SentiStrength_DataEnglishFeb2017/"};
		sentiStrength.initialise(ssthInitialisation); 
		
       
        
		for (int m = 0; m < data.M; m++){
			int N = data.docs[m].length;
			i[m] = new Vector<Integer>();
			s[m] = new Vector<Integer>();
			y[m] = new Vector<Integer>();
			
			//StringTokenizer tknr = new StringTokenizer(data.docs[m].rawStr, " \t\r\n");
			String [] tokens = data.docs[m].rawStr.split("[ \\t\\r\\n]");
			
			//initilize for e, t and s
			for (int n = 0; n < N; n++){
				
				//String token = tknr.nextToken();
				String token = tokens[n];
				
				Integer issue = null;
				Integer sentiment = null;
				Integer stance = null;
				//int mainSentiment = 0;
				//int longest = 0;
				
				int j = 0;
				for(List<String> issueList:initialIssueWord){
					
					for(String issueWord:issueList){
						if(token.equals(issueWord.toLowerCase())){
							issue = j;
							break;
						}
					}
					j++;
					if(issue != null){
						break;
					}
				}
				
				j = 0;
				for(List<String> stanceList:initialStanceWord){
					
					for(String stanceWord:stanceList){
						if(token.equals(stanceWord.toLowerCase())){
							stance = j;
							break;
						}
					}
					j++;
					if(stance != null){
						break;
					}
				}
				
				
				
				String[] words = sentiStrength.computeSentimentScores(token).split("\\s+");
				int pos = Integer.parseInt(words[0]);
				int neg = Integer.parseInt(words[1]);
				
				if (Math.abs(pos) < Math.abs(neg)) {
					sentiment = 1;
				}else if(Math.abs(pos) > Math.abs(neg)) {
					sentiment = 0;
				}
				
				
				
				if(issue == null){
					issue = (int)Math.floor(Math.random() * I);
				}
				if(sentiment == null){
					sentiment = (int)Math.floor(Math.random() * S);
				}
				if(stance == null){
					stance = (int)Math.floor(Math.random() * Y);
				}
				
				
				i[m].add(issue);
				s[m].add(sentiment);
				y[m].add(stance);
				
				nwi[data.docs[m].words[n]][issue] += 1;
				nws[data.docs[m].words[n]][sentiment] += 1;
				nwy[data.docs[m].words[n]][stance] += 1;
				nds[m][sentiment] += 1;
				ni[issue] += 1;
				ns[sentiment] += 1;
				ny[stance] += 1;
				ndi[m][issue] += 1;
				ndis[m][issue][sentiment] += 1;
				nd[m] += 1;
				ndisy[m][issue][sentiment][stance] += 1;
			}
			
			
			//System.out.println(e[m]);
			
		}
		
		
		
		theta_i = new double[M][I];
		theta_s = new double[M][S];
		theta_y = new double[M][I][S][Y];
		
		phi_i = new double[I][V];
		phi_s = new double[S][V];
		phi_y = new double[Y][V];
		
		//initialize beta_iw with value of beta
		beta_iw = new double[I][V];
		for(int i = 0; i < I; i++){
			for(int w = 0; w < V; w++){
				beta_iw[i][w] = beta_i;
			}
		}
		
		//initialize beta_sw with value of beta
		beta_sw=new double[S][V];
		for(int s = 0; s < S; s++){
			for(int w = 0; w < V; w++){
				beta_sw[s][w] = beta_s;
			}
		}
		
		//initialize beta_yw with value of beta
		beta_yw=new double[Y][V];
		for(int y = 0; y < Y; y++){
			for(int w = 0; w < V; w++){
				beta_yw[y][w] = beta_y;
			}
		}
				
		//Initialize lambda_e 
		lambda_i = new double[I][V];
		for(int i = 0; i < I; i++){
			for(int w = 0; w < V; w++){
				lambda_i[i][w] = 1;
			}
		}
		
		//Initialize lambda_s 
		lambda_s = new double[S][V];
		for(int s = 0; s < S; s++){
			for(int w = 0; w < V; w++){
				lambda_s[s][w] = 1;
			}
		}
		
		//Initialize lambda_y 
		lambda_y = new double[Y][V];
		for(int y = 0; y < Y; y++){
			for(int w = 0; w < V; w++){
				lambda_y[y][w] = 1;
			}
		}
				
		
		alpha_isy = new double[I][S][Y];
		alphaSum_is = new double[I][S];
		
		for(int i = 0; i < I; i++){
			for(int s = 0; s < S; s++){
				alphaSum_is[i][s] = 0;
				for(int y = 0; y < Y; y++){
					alpha_isy[i][s][y] = alpha_y;
					alphaSum_is[i][s] += alpha_isy[i][s][y];
				}
			}
			
			
		}
		
		
		prior2beta();
		
		

		beta_isum=new double[I];
		for(int i = 0; i < I; i++){
			for(int w = 0; w < V; w++){
				beta_isum[i] += beta_iw[i][w];
			}
		}
		
		beta_ssum=new double[S];
		for(int s = 0; s < S; s++){
			for(int w = 0; w < V; w++){
				beta_ssum[s] += beta_sw[s][w];
			}
		}
		
		beta_ysum=new double[Y];
		for(int y = 0; y < Y; y++){
			for(int w = 0; w < V; w++){
				beta_ysum[y] += beta_yw[y][w];
			}
		}
		return true;
	}
	
	/**
	 * initialize the model
	 */
	protected boolean init(LDACmdOption option){		
		if (option == null)
			return false;
		
		modelName = option.modelName;
		I = option.I;
		S = option.S;
		Y = option.Y;
				
		alpha_i = option.alpha_i;
		alpha_s = option.alpha_s;
		alpha_y = option.alpha_y;
		
		
		if (alpha_i < 0.0)
			alpha_i = 50.0 / I;
		
		if (alpha_s < 0.0)
			alpha_s = 50.0 / S;
		

		if (alpha_y < 0.0)
			alpha_y = 50.0 / Y;
		
		if (option.beta_i >= 0)
			beta_i = option.beta_i;
		
		if (option.beta_s >= 0)
			beta_s = option.beta_s;
		
		if (option.beta_y >= 0)
			beta_y = option.beta_y;
		
		niters = option.niters;
		
		dir = option.dir;
		if (dir.endsWith(File.separator))
			dir = dir.substring(0, dir.length() - 1);
		
		dfile = option.dfile;
		twords = option.twords;
		wordMapFile = option.wordMapFileName;
		
		return true;
	}
	
	public void loadData() {
		
		
		for (int i = 1; i <= I; i++) {
			
			initialIssueWord.add(loadPriorData("topicList/issues/"+i+".txt"));
				
				
			
		}
		
		for (int i = 1; i <= Y; i++) {
			initialStanceWord.add(loadPriorData("topicList/stance/"+i+".txt"));
		}
		
	}

	public List<String> loadPriorData(String topiclist){
		List topics = new ArrayList();
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(topiclist), "UTF-8"));
			String line = null;
			while ((line = reader.readLine()) != null) {
				topics.add(line);
				
			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return topics;
		
	}
	
	

	public void prior2beta(){
		Map<String, Integer> wordmap = data.localDict.word2id;
		Iterator<String> it1 = wordmap.keySet().iterator();
		Iterator<String> it2 = wordmap.keySet().iterator();
		Iterator<String> it3 = wordmap.keySet().iterator();
			
		
		//update lambda_i
		while (it1.hasNext()){
			String key = it1.next();
			Integer value = wordmap.get(key);
			for(int i = 0; i < I; i++){
				if(initialIssueWord.get(i).contains(key)){
					for(int j = 0; j < I; j++){
						if(j != i){
							lambda_i[j][value] = 0;
						}else {
							
						}
					}
					
				}
			}
			
			
		}
		
		//update lambda_y
		while (it2.hasNext()){
			String key = it2.next();
			Integer value = wordmap.get(key);
			for(int y = 0; y < Y; y++){
				if(initialStanceWord.get(y).contains(key)){
					for(int j = 0; j < Y; j++){
						if(j != y){
							lambda_y[j][value] = 0;
						}else {
									
						}
					}
							
				}
			}
					
					
		}
		
	
		
		
		SentiStrength sentiStrength = new SentiStrength(); 
		String ssthInitialisation[] = {"sentidata", "topicList/sentiment/SentiStrength_DataEnglishFeb2017/"};
		sentiStrength.initialise(ssthInitialisation); 
		while (it3.hasNext()){
			String key = it3.next();
			Integer value = wordmap.get(key);
			String[] words = sentiStrength.computeSentimentScores(key).split("\\s+");
			int pos = Integer.parseInt(words[0]);
			int neg = Integer.parseInt(words[1]);
			Integer sentiment = null;
			int strength = 0;
			if (Math.abs(pos) > Math.abs(neg)) {
				sentiment = 0;
				strength = pos;
			}else if (Math.abs(pos) < Math.abs(neg)) {
				sentiment = 1;
				strength = neg;
			}
			if (sentiment != null) {
				for(int j = 0; j < S; j++){
					if( j != sentiment) {
						lambda_s[j][value] = 0.0;
					} else if (Math.abs(pos+neg) == 4) {
						lambda_s[j][value] = 0.80;
					} else if (Math.abs(pos+neg) == 3) {
						lambda_s[j][value] = 0.60;
					} else if (Math.abs(pos+neg) == 2) {
						lambda_s[j][value] = 0.40;
					} else if (Math.abs(pos+neg) == 1) {
						lambda_s[j][value] = 0.20;
					} 
					
				}
			} 
		}
		
		
		//lambda_e x beta_ew
		for(int i = 0; i < I; i++){
			for(int w = 0; w < V; w++){
				beta_iw[i][w] = beta_iw[i][w] * lambda_i[i][w];
				
			}
			
		}
		
		//lambda_s x beta_sw
		for(int s = 0; s < S; s++){
			for(int w = 0; w < V; w++){
				beta_sw[s][w] = beta_sw[s][w] * lambda_s[s][w];
				
			}
			
		}
		

		//lambda_y x beta_yw
		for(int y = 0; y < Y; y++){
			for(int w = 0; w < V; w++){
				beta_yw[y][w] = beta_yw[y][w] * lambda_y[y][w];
				
			}
			
		}
		
		
	}

	/**
	 * init parameter for continue estimating or for later inference
	 */
	public boolean initEstimatedModel(LDACmdOption option){
		if (!init(option))
			return false;
		
			
		p = new double[I][S][Y];
		
		// load model, i.e., read z and trndata
		if (!loadModel()){
			System.out.println("Fail to load word-topic assignment file of the model!\n");
			return false;
		}
		
		
		
		System.out.println("Model loaded:");
		System.out.println("\talpha_i:" + alpha_i);
		System.out.println("\talpha_s:" + alpha_s);
		System.out.println("\talpha_y:" + alpha_y);
		System.out.println("\tbeta_i:" + beta_i);
		System.out.println("\tbeta_s:" + beta_s);
		System.out.println("\tbeta_y:" + beta_y);
		System.out.println("\tM:" + M);
		System.out.println("\tV:" + V);		
		
		nwi = new int[V][I];
		for (int w = 0; w < V; w++){
			for (int i = 0; i < I; i++){
				nwi[w][i] = 0;
			}
		}
		
		nws = new int[V][S];
		for (int w = 0; w < V; w++){
			for (int s = 0; s < S; s++){
				nws[w][s] = 0;
			}
		}
		
		nwy = new int[V][Y];
		for (int w = 0; w < V; w++){
			for (int y = 0; y < Y; y++){
				nwy[w][y] = 0;
			}
		}
		
		ni =  new int[I];
		for (int i = 0; i < I; i++) {
			ni[i] = 0;
		}
		
		ns =  new int[S];
		for (int s = 0; s < S; s++) {
			ns[s] = 0;
		}
		
		ny =  new int[Y];
		for (int y = 0; y < Y; y++) {
			ny[y] = 0;
		}
		
		ndi = new int[M][I];
		for (int m = 0; m < M; m++){
			for (int i = 0; i < I; i++) {
				ndi[m][i] = 0;
			}
		}
		

		nds = new int[M][S];
		for (int m = 0; m < M; m++){
			for (int s = 0; s < S; s++) {
				nds[m][s] = 0;
			}
		}
		
		nd = new int[M];
		for (int m = 0; m < M; m++){
			nd[m] = 0;
			
		}
		
		ndis =  new int[M][I][S];
		for (int m = 0; m < M; m++){
			for (int i = 0; i < I; i++) {
				for (int s = 0; s < S; s++) {
					ndis[m][i][s] = 0;
				}
			}
		}
		
		ndisy =  new int[M][I][S][Y];
		for (int m = 0; m < M; m++){
			for (int i = 0; i < I; i++) {
				for (int s = 0; s < S; s++) {
					for (int y = 0; y < Y; y++) {
						ndisy[m][i][s][y] = 0;
					}					
				}
			}
		}
		
			    
	    for (int m = 0; m < data.M; m++){
	    	int N = data.docs[m].length;
	    	
	    	for (int n = 0; n < N; n++){
	    		int w = data.docs[m].words[n];
	    		int issue = (Integer)i[m].get(n);
	    		int sentiment = (Integer)s[m].get(n);
	    		int stance = (Integer)y[m].get(n);
	    		
	    		nwi[data.docs[m].words[n]][issue] += 1;
				nws[data.docs[m].words[n]][sentiment] += 1;
				nwy[data.docs[m].words[n]][stance] += 1;
				nds[m][sentiment] += 1;
				ni[issue] += 1;
				ns[sentiment] += 1;
				ny[stance] += 1;
				ndi[m][issue] += 1;
				ndis[m][issue][sentiment] += 1;
				nd[m] += 1;
				ndisy[m][issue][sentiment][stance] += 1; 		
	    	}
	    	
	    }
	    
	    theta_i = new double[M][I];
		theta_s = new double[M][S];
		theta_y = new double[M][I][S][Y];
		
		phi_i = new double[I][V];
		phi_s = new double[S][V];
		phi_y = new double[Y][V];
		
		//initialize beta_iw with value of beta
		beta_iw = new double[I][V];
		for(int i = 0; i < I; i++){
			for(int w = 0; w < V; w++){
				beta_iw[i][w] = beta_i;
			}
		}
		
		//initialize beta_sw with value of beta
		beta_sw = new double[S][V];
		for(int s = 0; s < S; s++){
			for(int w = 0; w < V; w++){
				beta_sw[s][w] = beta_s;
			}
		}
		
		//initialize beta_yw with value of beta
		beta_yw = new double[Y][V];
		for(int y = 0; y < Y; y++){
			for(int w = 0; w < V; w++){
				beta_yw[y][w] = beta_y;
			}
		}
		
		alpha_isy = new double[I][S][Y];
		alphaSum_is = new double[I][S];
		
		for(int i = 0; i < I; i++){
			for(int s = 0; s < S; s++){
				alphaSum_is[i][s] = 0;
				for(int y = 0; y < Y; y++){
					alpha_isy[i][s][y] = alpha_y;
					alphaSum_is[i][s] += alpha_isy[i][s][y];
				}
			}
			
			
		}
		
				
		//Initialize lambda_e 
		lambda_i = new double[I][V];
		for(int i = 0; i < I; i++){
			for(int w = 0; w < V; w++){
				lambda_i[i][w] = 1;
			}
		}
		
		//Initialize lambda_s 
		lambda_s = new double[S][V];
		for(int s = 0; s < S; s++){
			for(int w = 0; w < V; w++){
				lambda_s[s][w] = 1;
			}
		}
		
		//Initialize lambda_y 
		lambda_y = new double[Y][V];
		for(int y = 0; y < Y; y++){
			for(int w = 0; w < V; w++){
				lambda_y[y][w] = 1;
			}
		}
		
		loadData();
		prior2beta();
		
		

		beta_isum = new double[I];
		for(int i = 0; i < I; i++){
			for(int w = 0; w < V; w++){
				beta_isum[i] += beta_iw[i][w];
			}
		}
		
		beta_ssum = new double[S];
		for(int s = 0; s < S; s++){
			for(int w = 0; w < V; w++){
				beta_ssum[s] += beta_sw[s][w];
			}
		}
		
		beta_ysum = new double[Y];
		for(int y = 0; y < Y; y++){
			for(int w = 0; w < V; w++){
				beta_ysum[y] += beta_yw[y][w];
			}
		}
		
	    
		

	    
		dir = option.dir;
		savestep = option.savestep;
		
		return true;
	}
	
	/**
	 * load saved model
	 */
	public boolean loadModel(){
		if (!readOthersFile(dir + File.separator + modelName + othersSuffix))
			return false;
		
		if (!readTAssignFile(dir + File.separator + modelName + tassignSuffix))
			return false;
		
		// read dictionary
		Dictionary dict = new Dictionary();
		if (!dict.readWordMap(dir + File.separator + wordMapFile))
			return false;
			
		data.localDict = dict;
		
		return true;
	}
	
	/**
	 * read other file to get parameters
	 */
	protected boolean readOthersFile(String otherFile){
		//open file <model>.others to read:
		
		try {
			BufferedReader reader = new BufferedReader(new FileReader(otherFile));
			String line;
			while((line = reader.readLine()) != null){
				StringTokenizer tknr = new StringTokenizer(line,"= \t\r\n");
				
				int count = tknr.countTokens();
				if (count != 2)
					continue;
				
				String optstr = tknr.nextToken();
				String optval = tknr.nextToken();
				
				if (optstr.equalsIgnoreCase("alpha_i")){
					alpha_i = Double.parseDouble(optval);					
				}
				else if (optstr.equalsIgnoreCase("alpha_s")){
					alpha_s = Double.parseDouble(optval);					
				}
				else if (optstr.equalsIgnoreCase("alpha_y")){
					alpha_y = Double.parseDouble(optval);					
				}
				else if (optstr.equalsIgnoreCase("beta_i")){
					beta_i = Double.parseDouble(optval);
				}
				else if (optstr.equalsIgnoreCase("beta_s")){
					beta_s = Double.parseDouble(optval);
				}
				else if (optstr.equalsIgnoreCase("beta_y")){
					beta_y = Double.parseDouble(optval);
				}
				else if (optstr.equalsIgnoreCase("itopics")){
					I = Integer.parseInt(optval);
				}
				else if (optstr.equalsIgnoreCase("stopics")){
					S = Integer.parseInt(optval);
				}
				else if (optstr.equalsIgnoreCase("ytopics")){
					Y = Integer.parseInt(optval);
				}
				else if (optstr.equalsIgnoreCase("liter")){
					liter = Integer.parseInt(optval);
				}
				else if (optstr.equalsIgnoreCase("nwords")){
					V = Integer.parseInt(optval);
				}
				else if (optstr.equalsIgnoreCase("ndocs")){
					M = Integer.parseInt(optval);
				}
				else {
					// any more?
				}
			}
			
			reader.close();
		}
		catch (Exception e){
			System.out.println("Error while reading other file:" + e.getMessage());
			e.printStackTrace();
			return false;
		}
		return true;
	}
	
	protected boolean readTAssignFile(String tassignFile){
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(
					new FileInputStream(tassignFile), "UTF-8"));
			
			String line;
			i = new Vector[M];	
			s = new Vector[M];
			y = new Vector[M];
			
			
			data = new LDADataset(M);
			data.V = V;			
			for (int k = 0; k < M; k++){
				line = reader.readLine();
				StringTokenizer tknr = new StringTokenizer(line, " \t\r\n");
				
				int length = tknr.countTokens();
				
				Vector<Integer> words = new Vector<Integer>();
				Vector<Integer> issues= new Vector<Integer>();
				Vector<Integer> sentiment = new Vector<Integer>();
				Vector<Integer> stance = new Vector<Integer>();
				
				for (int j = 0; j < length; j++){
					String token = tknr.nextToken();
					
					StringTokenizer tknr2 = new StringTokenizer(token, ":");
					if (tknr2.countTokens() != 4){
						System.out.println("Invalid word-topic assignment line\n");
						return false;
					}
					
					words.add(Integer.parseInt(tknr2.nextToken()));
					issues.add(Integer.parseInt(tknr2.nextToken()));
					sentiment.add(Integer.parseInt(tknr2.nextToken()));
					stance.add(Integer.parseInt(tknr2.nextToken()));
				}//end for each topic assignment
				
				
				//allocate and add new document to the corpus
				Document doc = new Document(words);
				data.setDoc(doc, k);
				
				//assign values for i
				i[k] = new Vector<Integer>();
				for (int j = 0; j < issues.size(); j++){
					i[k].add(issues.get(j));
				}
				
							
				//assign values for s
				s[k] = new Vector<Integer>();
				for (int j = 0; j < sentiment.size(); j++){
					s[k].add(sentiment.get(j));
				}
				
				//assign values for y
				y[k] = new Vector<Integer>();
				for (int j = 0; j < stance.size(); j++){
					y[k].add(stance.get(j));
				}
			}//end for each doc
			
			reader.close();
		}
		catch (Exception e){
			System.out.println("Error while loading model: " + e.getMessage());
			e.printStackTrace();
			return false;
		}
		return true;
	}
	
	/**
	 * Save model
	 */
	public boolean saveModel(String modelName){
		if (!saveModelTAssign(dir + File.separator + modelName + tassignSuffix)){
			return false;
		}
		
		if (!saveModelOthers(dir + File.separator + modelName + othersSuffix)){			
			return false;
		}
		
		if (!saveModelThetaI(dir + File.separator + modelName + "_I" +thetaSuffix)){
			return false;
		}
		
		if (!saveModelThetaS(dir + File.separator + modelName + "_S" +thetaSuffix)){
			return false;
		}
		
		if (!saveModelThetaY(dir + File.separator + modelName + "_Y" +thetaSuffix)){
			return false;
		}
		
		if (!saveModelPhiI(dir + File.separator + modelName + "_I" + phiSuffix)){
			return false;
		}
		
		if (!saveModelPhiS(dir + File.separator + modelName + "_S" + phiSuffix)){
			return false;
		}
		
		if (!saveModelPhiY(dir + File.separator + modelName + "_Y" + phiSuffix)){
			return false;
		}
		
		if (twords > 0){
			if (!saveModelIwords(dir + File.separator + modelName + "_I"+ twordsSuffix))
				return false;
		}
		
		if (twords > 0){
			if (!saveModelSwords(dir + File.separator + modelName + "_S"+ twordsSuffix))
				return false;
		}
		
		if (twords > 0){
			if (!saveModelYwords(dir + File.separator + modelName + "_Y"+ twordsSuffix))
				return false;
		}
		return true;
	}
	
	/**
	 * Save word-topic assignments for this model
	 */
	public boolean saveModelTAssign(String filename){
		int k, j;
		
		try{
			BufferedWriter writer = new BufferedWriter(new FileWriter(filename));
			
			//write docs with topic assignments for words
			for (k = 0; k < data.M; k++){
				for (j = 0; j < data.docs[k].length; ++j){
					writer.write(data.docs[k].words[j] + ":" + i[k].get(j) + ":" + s[k].get(j) + ":" + y[k].get(j) + " ");					
				}
				writer.write("\n");
			}
				
			writer.close();
		}
		catch (Exception e){
			System.out.println("Error while saving model tassign: " + e.getMessage());
			e.printStackTrace();
			return false;
		}
		return true;
	}
	
	/**
	 * Save other information of this model
	 */
	public boolean saveModelOthers(String filename){
		try{
			BufferedWriter writer = new BufferedWriter(new FileWriter(filename));
			
			writer.write("alpha_i=" + alpha_i + "\n");
			writer.write("alpha_s=" + alpha_s + "\n");
			writer.write("alpha_y=" + alpha_y + "\n");
			writer.write("beta_i=" + beta_i + "\n");
			writer.write("beta_s=" + beta_s + "\n");
			writer.write("beta_y=" + beta_y + "\n");
			writer.write("itopics=" + I + "\n");
			writer.write("stopics=" + S + "\n");
			writer.write("ytopics=" + Y + "\n");
			writer.write("ndocs=" + M + "\n");
			writer.write("nwords=" + V + "\n");
			writer.write("liters=" + liter + "\n");
			
			writer.close();
		}
		catch(Exception e){
			System.out.println("Error while saving model others:" + e.getMessage());
			e.printStackTrace();
			return false;
		}
		return true;
	}
	
	/**
	 * Save theta (Issue distribution) for this model
	 */
	public boolean saveModelThetaI(String filename){
		try{
			BufferedWriter writer = new BufferedWriter(new FileWriter(filename));
			for (int m = 0; m < M; m++){
				for (int i = 0; i < I; i++){
					writer.write(theta_i[m][i] + " ");
				}
				writer.write("\n");
			}
			writer.close();
		}
		catch (Exception e){
			System.out.println("Error while saving topic distribution file for this model: " + e.getMessage());
			e.printStackTrace();
			return false;
		}
		return true;
	}
	
	/**
	 * Save theta (sentiment distribution) for this model
	 */
	public boolean saveModelThetaS(String filename){
		try{
			BufferedWriter writer = new BufferedWriter(new FileWriter(filename));
			for (int m = 0; m < M; m++){
				for (int s = 0; s < S; s++){
					writer.write(theta_s[m][s] + " ");
				}
				writer.write("\n");
			}
			writer.close();
		}
		catch (Exception e){
			System.out.println("Error while saving topic distribution file for this model: " + e.getMessage());
			e.printStackTrace();
			return false;
		}
		return true;
	}
	
	/**
	 * Save theta (stance distribution) for this model
	 */
	public boolean saveModelThetaY(String filename){
		try{
			BufferedWriter writer = new BufferedWriter(new FileWriter(filename));
			for (int m = 0; m < M; m++){
				for(int i = 0; i < I; i++) {
					for (int s = 0; s < S; s++){
						for(int y = 0; y < Y; y++) {
							writer.write(theta_y[m][i][s][y] + " ");
						}						
					}
				}
				
				writer.write("\n");
			}
			writer.close();
		}
		catch (Exception e){
			System.out.println("Error while saving topic distribution file for this model: " + e.getMessage());
			e.printStackTrace();
			return false;
		}
		return true;
	}
	
	
	/**
	 * Save word-entity distribution
	 */
	
	public boolean saveModelPhiI(String filename){
		try {
			BufferedWriter writer = new BufferedWriter(new FileWriter(filename));
			
			for (int i = 0; i < I; i++){
				for (int j = 0; j < V; j++){
					writer.write(phi_i[i][j] + " ");
				}
				writer.write("\n");
			}
			writer.close();
		}
		catch (Exception e){
			System.out.println("Error while saving word-topic distribution:" + e.getMessage());
			e.printStackTrace();
			return false;
		}
		return true;
	}
	
	/**
	 * Save word-sentiment distribution
	 */
	
	public boolean saveModelPhiS(String filename){
		try {
			BufferedWriter writer = new BufferedWriter(new FileWriter(filename));
			
			for (int s = 0; s < S; s++){
				for (int j = 0; j < V; j++){
					writer.write(phi_s[s][j] + " ");
				}
				writer.write("\n");
			}
			writer.close();
		}
		catch (Exception e){
			System.out.println("Error while saving word-topic distribution:" + e.getMessage());
			e.printStackTrace();
			return false;
		}
		return true;
	}
	
	/**
	 * Save word-stance distribution
	 */
	
	public boolean saveModelPhiY(String filename){
		try {
			BufferedWriter writer = new BufferedWriter(new FileWriter(filename));
			
			for (int y = 0; y < Y; y++){
				for (int j = 0; j < V; j++){
					writer.write(phi_y[y][j] + " ");
				}
				writer.write("\n");
			}
			writer.close();
		}
		catch (Exception e){
			System.out.println("Error while saving word-topic distribution:" + e.getMessage());
			e.printStackTrace();
			return false;
		}
		return true;
	}
	
	/**
	 * Save model the most likely words for each entity
	 */
	public boolean saveModelIwords(String filename){
		try{
			BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(
					new FileOutputStream(filename), "UTF-8"));
			
			if (twords > V){
				twords = V;
			}
			
			for (int i = 0; i < I; i++){
				List<Pair> wordsProbsList = new ArrayList<Pair>(); 
				for (int w = 0; w < V; w++){
					Pair p = new Pair(w, phi_i[i][w], false);
					
					wordsProbsList.add(p);
				}//end foreach word
				
				//print topic				
				writer.write("Issue " + i + "th:\n");
				Collections.sort(wordsProbsList);
				
				for (int w = 0; w < twords; w++){
					if (data.localDict.contains((Integer)wordsProbsList.get(w).first)){
						String word = data.localDict.getWord((Integer)wordsProbsList.get(w).first);
						
						writer.write("\t" + word + " " + wordsProbsList.get(w).second + "\n");
					}
				}
			} //end foreach topic			
						
			writer.close();
		}
		catch(Exception e){
			System.out.println("Error while saving model twords: " + e.getMessage());
			e.printStackTrace();
			return false;
		}
		return true;
	}
	
	
	/**
	 * Save model the most likely words for each entity
	 */
	public boolean saveModelSwords(String filename){
		try{
			BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(
					new FileOutputStream(filename), "UTF-8"));
			
			if (twords > V){
				twords = V;
			}
			
			for (int s = 0; s < S; s++){
				List<Pair> wordsProbsList = new ArrayList<Pair>(); 
				for (int w = 0; w < V; w++){
					Pair p = new Pair(w, phi_s[s][w], false);
					
					wordsProbsList.add(p);
				}//end foreach word
				
				//print topic				
				writer.write("Sentiment " + s + "th:\n");
				Collections.sort(wordsProbsList);
				
				for (int i = 0; i < twords; i++){
					if (data.localDict.contains((Integer)wordsProbsList.get(i).first)){
						String word = data.localDict.getWord((Integer)wordsProbsList.get(i).first);
						
						writer.write("\t" + word + " " + wordsProbsList.get(i).second + "\n");
					}
				}
			} //end foreach topic			
						
			writer.close();
		}
		catch(Exception e){
			System.out.println("Error while saving model twords: " + e.getMessage());
			e.printStackTrace();
			return false;
		}
		return true;
	}
	
	/**
	 * Save model the most likely words for each entity
	 */
	public boolean saveModelYwords(String filename){
		try{
			BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(
					new FileOutputStream(filename), "UTF-8"));
			
			if (twords > V){
				twords = V;
			}
			
			for (int y = 0; y < Y; y++){
				List<Pair> wordsProbsList = new ArrayList<Pair>(); 
				for (int w = 0; w < V; w++){
					Pair p = new Pair(w, phi_y[y][w], false);
					
					wordsProbsList.add(p);
				}//end foreach word
				
				//print topic				
				writer.write("Stance " + y + "th:\n");
				Collections.sort(wordsProbsList);
				
				for (int i = 0; i < twords; i++){
					if (data.localDict.contains((Integer)wordsProbsList.get(i).first)){
						String word = data.localDict.getWord((Integer)wordsProbsList.get(i).first);
						
						writer.write("\t" + word + " " + wordsProbsList.get(i).second + "\n");
					}
				}
			} //end foreach topic			
						
			writer.close();
		}
		catch(Exception e){
			System.out.println("Error while saving model twords: " + e.getMessage());
			e.printStackTrace();
			return false;
		}
		return true;
	}


	/**
	 * Init parameters for inference
	 * reading new dataset from file
	 */
	public boolean initNewModel(LDACmdOption option, Model trnModel){
		if (!init(option))
			return false;
		
		LDADataset dataset = LDADataset.readDataSet(dfile, trnModel.data.localDict, dir);
		if (dataset == null){
			System.out.println("Fail to read dataset!\n");
			return false;
		}
		
		return initNewModel(option, dataset , trnModel);
	}
	

	/**
	 * Init parameters for inference
	 * @param newData DataSet for which we do inference
	 */
	public boolean initNewModel(LDACmdOption option, LDADataset newData, Model trnModel){
		if (!init(option))
			return false;
		
				
		I = trnModel.I;
		S = trnModel.S;
		Y = trnModel.Y;
		
		alpha_i = trnModel.alpha_i;
		alpha_s = trnModel.alpha_s;
		alpha_y = trnModel.alpha_y;
		beta_i = trnModel.beta_i;
		beta_s = trnModel.beta_s;
		beta_y = trnModel.beta_y;
				
		p = new double[I][S][Y];
				
		data = newData;
		
		//+ allocate memory and assign values for variables		
		M = data.M;
		V = data.V;
		dir = option.dir;
		savestep = option.savestep;
		System.out.println("M:" + M);
		System.out.println("V:" + V);
		
		// K: from command line or default value
	    // alpha, beta: from command line or default values
	    // niters, savestep: from command line or default values
		nwi = new int[V][I];
		for (int w = 0; w < V; w++){
			for (int i = 0; i < I; i++){
				nwi[w][i] = 0;
			}
		}
		
		nws = new int[V][S];
		for (int w = 0; w < V; w++){
			for (int s = 0; s < S; s++){
				nws[w][s] = 0;
			}
		}
		
		nwy = new int[V][Y];
		for (int w = 0; w < V; w++){
			for (int y = 0; y < Y; y++){
				nwy[w][y] = 0;
			}
		}
		
		ni =  new int[I];
		for (int i = 0; i < I; i++) {
			ni[i] = 0;
		}
		
		ns =  new int[S];
		for (int s = 0; s < S; s++) {
			ns[s] = 0;
		}
		
		ny =  new int[Y];
		for (int y = 0; y < Y; y++) {
			ny[y] = 0;
		}
		
		ndi = new int[M][I];
		for (int m = 0; m < M; m++){
			for (int i = 0; i < I; i++) {
				ndi[m][i] = 0;
			}
		}
		

		nds = new int[M][S];
		for (int m = 0; m < M; m++){
			for (int s = 0; s < S; s++) {
				nds[m][s] = 0;
			}
		}
		
		nd = new int[M];
		for (int m = 0; m < M; m++){
			nd[m] = 0;
			
		}
		
		ndis =  new int[M][I][S];
		for (int m = 0; m < M; m++){
			for (int i = 0; i < I; i++) {
				for (int s = 0; s < S; s++) {
					ndis[m][i][s] = 0;
				}
			}
		}
		
		ndisy =  new int[M][I][S][Y];
		for (int m = 0; m < M; m++){
			for (int i = 0; i < I; i++) {
				for (int s = 0; s < S; s++) {
					for (int y = 0; y < Y; y++) {
						ndisy[m][i][s][y] = 0;
					}					
				}
			}
		}
		
		loadData();
		
		i = new Vector[M];
		s = new Vector[M];
		y = new Vector[M];
		
		SentiStrength sentiStrength = new SentiStrength(); 
		String ssthInitialisation[] = {"sentidata", "topicList/sentiment/SentiStrength_DataEnglishFeb2017/"};
		sentiStrength.initialise(ssthInitialisation); 
		
		for (int m = 0; m < data.M; m++){
			int N = data.docs[m].length;
			i[m] = new Vector<Integer>();
			s[m] = new Vector<Integer>();
			y[m] = new Vector<Integer>();
			
			StringTokenizer tknr = new StringTokenizer(data.docs[m].rawStr, " \t\r\n");
			
			//initilize for e, t and s
			for (int n = 0; n < N; n++){
				String token = tknr.nextToken();
				Integer issue = null;
				Integer sentiment = null;
				Integer stance = null;
				
				int j = 0;
				for(List<String> issueList:initialIssueWord){
					
					for(String issueWord:issueList){
						if(token.equals(issueWord.toLowerCase())){
							issue = j;
							break;
						}
					}
					j++;
					if(issue != null){
						break;
					}
				}
				
				j = 0;
				for(List<String> stanceList:initialStanceWord){
					
					for(String stanceWord:stanceList){
						if(token.equals(stanceWord.toLowerCase())){
							issue = j;
							break;
						}
					}
					j++;
					if(issue != null){
						break;
					}
				}
				
				
				String[] words = sentiStrength.computeSentimentScores(token).split("\\s+");
				int pos = Integer.parseInt(words[0]);
				int neg = Integer.parseInt(words[1]);
				
				if (Math.abs(pos) < Math.abs(neg)) {
					sentiment = 1;
				}else if(Math.abs(pos) > Math.abs(neg)) {
					sentiment = 0;
				}
				
				
				if(issue == null){
					issue = (int)Math.floor(Math.random() * I);
				}
				if(sentiment == null){
					sentiment = (int)Math.floor(Math.random() * S);
				}
				if(stance == null){
					stance = (int)Math.floor(Math.random() * Y);
				}
				
				i[m].add(issue);
				s[m].add(sentiment);
				
				nwi[data.docs[m].words[n]][issue] += 1;
				nws[data.docs[m].words[n]][sentiment] += 1;
				nwy[data.docs[m].words[n]][stance] += 1;
				nds[m][sentiment] += 1;
				ni[issue] += 1;
				ns[sentiment] += 1;
				ny[stance] += 1;
				ndi[m][issue] += 1;
				ndis[m][issue][sentiment] += 1;
				nd[m] += 1;
				ndisy[m][issue][sentiment][stance] += 1;
			}
			
			
			
			
		}
		
		
		theta_i = new double[M][I];
		theta_s = new double[M][S];
		theta_y = new double[M][I][S][Y];
		
		phi_i = new double[I][V];
		phi_s = new double[S][V];
		phi_y = new double[Y][V];		
		
		//initialize beta_ew with value of beta
		beta_iw=new double[I][V];
		for(int i = 0; i < I; i++){
			for(int w = 0; w < V; w++){
				beta_iw[i][w] = beta_i;
			}
		}
		
		//initialize beta_ew with value of beta
		beta_sw=new double[S][V];
		for(int s = 0; s < S; s++){
			for(int w = 0; w < V; w++){
				beta_sw[s][w] = beta_s;
			}
		}
		
		//initialize beta_yw with value of beta
		beta_yw=new double[Y][V];
		for(int y = 0; y < Y; y++){
			for(int w = 0; w < V; w++){
				beta_yw[y][w] = beta_y;
			}
		}
		
		alpha_isy = new double[I][S][Y];
		alphaSum_is = new double[I][S];
		
		for(int i = 0; i < I; i++){
			for(int s = 0; s < S; s++){
				alphaSum_is[i][s] = 0;
				for(int y = 0; y < Y; y++){
					alpha_isy[i][s][y] = alpha_y;
					alphaSum_is[i][s] += alpha_isy[i][s][y];
				}
			}
			
			
		}
		
				
		//Initialize lambda_e 
		lambda_i = new double[I][V];
		for(int i = 0; i < I; i++){
			for(int w = 0; w < V; w++){
				lambda_i[i][w] = 1;
			}
		}
		
		//Initialize lambda_s 
		lambda_s = new double[S][V];
		for(int s = 0; s < S; s++){
			for(int w = 0; w < V; w++){
				lambda_s[s][w] = 1;
			}
		}
		
		//Initialize lambda_s 
		lambda_y = new double[Y][V];
		for(int y = 0; y < S; y++){
			for(int w = 0; w < V; w++){
				lambda_y[y][w] = 1;
			}
		}
		
		
		
		prior2beta();
		
		

		beta_isum = new double[I];
		for(int i = 0; i < I; i++){
			for(int w = 0; w < V; w++){
				beta_isum[i] += beta_iw[i][w];
			}
		}
		
		beta_ssum = new double[S];
		for(int s = 0; s < S; s++){
			for(int w = 0; w < V; w++){
				beta_ssum[s] += beta_sw[s][w];
			}
		}	
		
		beta_ysum = new double[Y];
		for(int y = 0; y < Y; y++){
			for(int w = 0; w < V; w++){
				beta_ysum[y] += beta_yw[y][w];
			}
		}
		
		
		return true;
	}
	
	
}
