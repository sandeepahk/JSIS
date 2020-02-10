package jis;

import java.io.File;

import org.apache.commons.math3.special.Gamma;


public class Estimator {
	// output model
		protected Model trnModel;
		LDACmdOption option;
		
		public boolean init(LDACmdOption option){
			this.option = option;
			trnModel = new Model();
			
			if (option.est){
				if (!trnModel.initNewModel(option))
					return false;
				trnModel.data.localDict.writeWordMap(option.dir + File.separator + option.wordMapFileName);
			}
			else if (option.estc){
				if (!trnModel.initEstimatedModel(option))
					return false;
			}
			
			return true; 
		}
		
		public void estimate(){
			System.out.println("Sampling " + trnModel.niters + " iteration!");
			
			int lastIter = trnModel.liter;
			for (trnModel.liter = lastIter + 1; trnModel.liter < trnModel.niters + lastIter; trnModel.liter++){
				System.out.println("Iteration " + trnModel.liter + " ...");
				
				// for all z_i
				for (int m = 0; m < trnModel.M; m++){				
					for (int n = 0; n < trnModel.data.docs[m].length; n++){
						int [] results = new int [2];
						results = sampling(m, n);
						
						trnModel.i[m].set(n, results[0]);
						trnModel.s[m].set(n, results[1]);
					}// end for each word
					
				}// end for each document
				
				if (trnModel.updateParaSteps > 0 && trnModel.liter % trnModel.updateParaSteps == 0){
					//update_Prameters_Sentiment();
					
				}									
				
				if (option.savestep > 0){
					if (trnModel.liter % option.savestep == 0){
						System.out.println("Saving the model at iteration " + trnModel.liter + " ...");
						computeTheta();
						computePhi();
						trnModel.saveModel("model-" + Conversion.ZeroPad(trnModel.liter, 5));
					}
				}
			}// end iterations		
			
			System.out.println("Gibbs sampling completed!\n");
			System.out.println("Saving the final model!\n");
			computeTheta();
			computePhi();
			trnModel.liter--;
			trnModel.saveModel("model-final");
		}
		

		/**
		 * Do sampling
		 * @param m document number
		 * @param n word number
		 * @return topic id
		 */
		public int[] sampling(int m, int n){
			// remove z_i from the count variable
			int issue = trnModel.i[m].get(n);
			int sentiment = trnModel.s[m].get(n);
			int w = trnModel.data.docs[m].words[n];
			
			trnModel.nwi[w][issue] -= 1;
			trnModel.nws[w][sentiment] -= 1;
			trnModel.ni[issue] -= 1;
			trnModel.ns[sentiment] -= 1;
			trnModel.ndi[m][issue] -= 1;
			trnModel.ndis[m][issue][sentiment] -= 1;
			trnModel.nd[m] -= 1;
		
			
			
			double Ialpha = trnModel.alpha_i * trnModel.I;
						
			
			//do multinominal sampling via cumulative method
			for (int i = 0; i < trnModel.I; i++){
				for (int s = 0; s < trnModel.S; s++) {
					trnModel.p[i][s] = (trnModel.nwi[w][i] + trnModel.beta_iw[i][w])/(trnModel.ni[i] + trnModel.beta_isum[i]) *
								(trnModel.nws[w][s] + trnModel.beta_sw[s][w])/(trnModel.ns[s] + trnModel.beta_ssum[s]) *
								(trnModel.ndi[m][i] + trnModel.alpha_i)/(trnModel.nd[m] + Ialpha) *
								(trnModel.ndis[m][i][s] + trnModel.alpha_is[i][s])/(trnModel.ndi[m][i] + trnModel.alphaSum_i[i]);
						
				}
				
				
			}
			
		
			// cumulate multinomial parameters
			for (int i = 0; i < trnModel.I; i++){
				for (int s = 0; s < trnModel.S; s++) {
						if( s == 0) {
							if(i == 0) {
									continue;
							}
							else {
								trnModel.p[i][s] += trnModel.p[i - 1][trnModel.S - 1];
							}
							
							
							
							
						}
						else {
							
							trnModel.p[i][s] += trnModel.p[i][s - 1];
						}
						
				}
				
			}
			
			// scaled sample because of unnormalized p[]
			double u = Math.random() * trnModel.p[trnModel.I - 1][trnModel.S - 1];
			
			boolean loopbreak = false;
			for (issue = 0; issue < trnModel.I ; issue++){
				for (sentiment = 0; sentiment < trnModel.S; sentiment++) {
					if (trnModel.p[issue][sentiment] > u) {//sample topic w.r.t distribution p
						loopbreak = true;
						break;
					}
							
				}
					
					
				
				
				if(loopbreak) {
					break;
				}
			}
			
			/*
			if( entity == trnModel.E) entity--;
			if ( topic == trnModel.T) topic--;
			if ( sentiment == trnModel.S) sentiment--;*/
			// add newly estimated z_i to count variables
			
			if(issue == trnModel.I) {
				String [] tokens = trnModel.data.docs[m].rawStr.split("[ \\t\\r\\n]");
				System.out.println(m+" "+tokens[n-1]+" " +tokens[n]);
			}
			
			trnModel.nwi[w][issue] += 1;
			trnModel.nws[w][sentiment] += 1;
			trnModel.ni[issue] += 1;
			trnModel.ns[sentiment] += 1;
			trnModel.ndi[m][issue] += 1;
			trnModel.ndis[m][issue][sentiment] += 1;
			trnModel.nd[m] += 1;
			
			int [] results = new int [2]; 
			
			results[0] = issue;
			results[1] = sentiment;
			
			
	 		return results;
		}
		
		public void computeTheta(){
			for (int m = 0; m < trnModel.M; m++){
				for (int i = 0; i < trnModel.I; i++){
					trnModel.theta_i[m][i] = (trnModel.ndi[m][i] + trnModel.alpha_i) / (trnModel.nd[m] + trnModel.I * trnModel.alpha_i);
				}
			}
						
			for (int m = 0; m < trnModel.M; m++){
				for (int i = 0; i < trnModel.I; i++){
					for (int s = 0; s < trnModel.S; s++) {
						trnModel.theta_s[m][i][s] = (trnModel.ndis[m][i][s] + trnModel.alpha_is[i][s]) / (trnModel.ndi[m][i] + trnModel.alphaSum_i[i]);
					}
					
				}
			}
			
			
		}
		
		public void computePhi(){
			for (int i = 0; i < trnModel.I; i++){
				for (int w = 0; w < trnModel.V; w++){
					trnModel.phi_i[i][w] = (trnModel.nwi[w][i] + trnModel.beta_iw[i][w]) / (trnModel.ni[i] + trnModel.beta_isum[i]);
				}
			}
			
						
			for (int s = 0; s < trnModel.S; s++){
				for (int w = 0; w < trnModel.V; w++){
					trnModel.phi_s[s][w] = (trnModel.nws[w][s] + trnModel.beta_sw[s][w]) / (trnModel.ns[s] + trnModel.beta_ssum[s]);
				}
			}
		}
		
		public int update_Prameters_Sentiment(){
			int[][] data = new int[trnModel.S][trnModel.M];
			double[] alpha_temp = new double[trnModel.S];
			
			for (int s = 0; s < trnModel.S; s++){
				for (int m = 0; m < trnModel.M; m++){
					data[s][m] = 0;
				}
				
			}
			
			for (int s = 0; s < trnModel.S; s++){
				alpha_temp[s] = 0;
			}
			
			//update alpha
			for(int i = 0; i < trnModel.I; i++){
				
					
					for(int s = 0; s < trnModel.S; s++){
						for(int m = 0; m <trnModel.M; m++){
							
							data[s][m] += trnModel.ndis[m][i][s];
						}
					}
				
					for(int s = 0; s < trnModel.S; s++){
						alpha_temp[s] = trnModel.alpha_is[i][s];
					}
				
			}
			
			polya_fit_simple(data, alpha_temp, trnModel.S, trnModel.M);
			
			for(int i = 0; i < trnModel.I; i++){
				// update alpha
				trnModel.alphaSum_i[i] = 0.0;
				for (int s = 0; s < trnModel.S; s++) {
					trnModel.alpha_is[i][s] = alpha_temp[s];
					trnModel.alphaSum_i[i] += trnModel.alpha_is[i][s];
				}
				
			}
					
					
					
			
			return 0;
			
		}
		
		private int polya_fit_simple(int[][] data, double[] alpha, int _K, int _nSample ){
			int K = _K;
			int nSample = _nSample;
			int polya_iter = Integer.MAX_VALUE;
			
			
			double sum_alpha_old;
			double[] old_alpha = new double[K];
			double sum_g = 0; //sum_g = sum_digama(data[i][k] + old_alpha[k]),
			double sum_h = 0; // sum_h + sum_digama(data[i] + sum_alpha_old) , where data[i] = sum_data[i][k] for all k,
			double[] data_row_sum = new double[nSample]; // the sum of the counts of each data sample P = {P_1, P_2,...,P_k}
			boolean sat_state = false;
			int i, k, j;
			
			for (k = 0; k < K; k++){
				old_alpha[k] = 0;
			}
			
			for (i = 0; i < nSample; i++) {
				data_row_sum[i] = 0;
			}
			
			for (i = 0; i < nSample; i++) {
				for (k = 0; k < K; k++){
					
					data_row_sum[i] += data[k][i];
					
				}
				
				
			}
			
			//simplw fix point interation
			System.out.println("Optimising parameters....");
			for(i = 0; i < polya_iter; i++){
				sum_alpha_old = 0;
				// update old_alpha after each iteration
				for(j = 0; j < K; j++){
					old_alpha[j] = alpha[j];
				}
				
				//calculate sum_alpha_old
				for (j = 0; j < K; j++) {
					 sum_alpha_old += old_alpha[j];
				 }
				
				for (k = 0; k < K; k++) {
					 sum_g = 0;
					 sum_h = 0;
					 
					 // calculate sum_g[k]
					 for (j = 0; j < nSample; j++) {
						 sum_g += Gamma.digamma( data[k][j] + old_alpha[k]);
					 }
					 
					// calculate sum_h
					 for (j = 0; j < nSample; j++) {
						 sum_h += Gamma.digamma(data_row_sum[j] + sum_alpha_old);
					 }
					 
							
					// update alpha (new)
					double x = sum_g - nSample * Gamma.digamma(old_alpha[k]);
					double y = sum_h - nSample * Gamma.digamma(sum_alpha_old);
					
					alpha[k] = old_alpha[k] * (x / y);
					
					
					
					
					 
						
					 
				}
				
				 // terminate iteration ONLY if each dimension of {alpha_1, alpha_2, ... alpha_k} satisfy the termination criteria,
				 for (j = 0; j < K; j++) {
					 if (Math.abs( alpha[j] - old_alpha[j]) > 0.00000001 ) {
						 break;
					 }
						 
					 if ( j == K - 1) {
						 sat_state = true;
					 }
				 }
				 
				// check whether to terminate the whole iteration
					if(sat_state) {
						System.out.println("Terminated at iteration: " + i);
						break;
					}
					else if(i == polya_iter - 1) {
						
						System.out.println("Haven't converged! Terminated at iteration: " + (i + 1));
					}
						
	 

				
			}
			 for (k = 0; k < K; k++) {
				 System.out.print(alpha[k] + "\t");
				 
			 }
			
			
			System.out.println("Optimisation done!");

		    return 0;
		}
		
		

}
