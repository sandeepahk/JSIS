package jsis_reverse;

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
						int [] results = new int [3];
						results = sampling(m, n);
						
						trnModel.i[m].set(n, results[0]);
						trnModel.s[m].set(n, results[1]);
						trnModel.y[m].set(n, results[2]);
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
		 * @return sentiment id
		 */
		public int[] sampling(int m, int n){
			// remove z_i from the count variable
			int issue = trnModel.i[m].get(n);
			int sentiment = trnModel.s[m].get(n);
			int stance = trnModel.y[m].get(n);
			int w = trnModel.data.docs[m].words[n];
			
			trnModel.nwi[w][issue] -= 1;
			trnModel.nws[w][sentiment] -= 1;
			trnModel.nwy[w][stance] -= 1;
			trnModel.ni[issue] -= 1;
			trnModel.ns[sentiment] -= 1;
			trnModel.ny[stance] -= 1;
			trnModel.ndyi[m][stance][issue] -= 1;
			trnModel.ndys[m][stance][sentiment] -= 1;
			trnModel.ndy[m][stance] -= 1;
			trnModel.nd[m] -= 1;
		
			
			
			double Yalpha = trnModel.alpha_y * trnModel.Y;
			
		
			
			
			//do multinominal sampling via cumulative method
			for (int e = 0; e < trnModel.I; e++){
				for (int t = 0; t < trnModel.S; t++) {
					for (int s = 0; s < trnModel.Y; s++) {
						trnModel.p[e][t][s] = (trnModel.nwi[w][e] + trnModel.beta_iw[e][w])/(trnModel.ni[e] + trnModel.beta_isum[e]) *
								(trnModel.nws[w][t] + trnModel.beta_sw[t][w])/(trnModel.ns[t] + trnModel.beta_ssum[t]) *
								(trnModel.nwy[w][s] + trnModel.beta_yw[s][w])/(trnModel.ny[s] + trnModel.beta_ysum[s]) *
								(trnModel.ndyi[m][s][e] + trnModel.alpha_yi[s][e])/(trnModel.ndy[m][s] + trnModel.alphaSum_yi[s]) *
								(trnModel.ndys[m][s][t] + trnModel.alpha_ys[s][t])/(trnModel.ndy[m][s] + trnModel.alphaSum_ys[s]) *
								(trnModel.ndy[m][s] + trnModel.alpha_y)/(trnModel.nd[m] + Yalpha);
						
					}
				}
				
			}
			
		
			// cumulate multinomial parameters
			for (int e = 0; e < trnModel.I; e++){
				for (int t = 0; t < trnModel.S; t++) {
					for (int s = 0; s < trnModel.Y; s++) {
						if( s == 0) {
							if(t == 0) {
								if(e == 0) {
									continue;
								}
								else {
									trnModel.p[e][t][s] += trnModel.p[e-1][trnModel.S - 1][trnModel.Y -1];
								}
							}
							else {
								trnModel.p[e][t][s] += trnModel.p[e][t-1][trnModel.Y - 1];
							}
							
							
						}
						else {
							
							trnModel.p[e][t][s] += trnModel.p[e][t][s-1];
						}
						
					}
				}
			}
			
			// scaled sample because of unnormalized p[]
			double u = Math.random() * trnModel.p[trnModel.I - 1][trnModel.S - 1][trnModel.Y - 1];
			
			boolean loopbreak = false;
			for (issue = 0; issue < trnModel.I ; issue++){
				for (sentiment = 0; sentiment < trnModel.S ; sentiment++) {
					for (stance = 0; stance < trnModel.Y; stance++) {
						if (trnModel.p[issue][sentiment][stance] > u) {//sample sentiment w.r.t distribution p
							loopbreak = true;
							break;
						}
							
					}
					
					if(loopbreak) {
						break;
					}
				}
				
				if(loopbreak) {
					break;
				}
			}
			
			/*
			if( issue == trnModel.E) issue--;
			if ( sentiment == trnModel.T) sentiment--;
			if ( stance == trnModel.S) stance--;*/
			// add newly estimated z_i to count variables
			
			trnModel.nwi[w][issue] += 1;
			trnModel.nws[w][sentiment] += 1;
			trnModel.nwy[w][stance] += 1;
			trnModel.ni[issue] += 1;
			trnModel.ns[sentiment] += 1;
			trnModel.ny[stance] += 1;
			trnModel.ndyi[m][stance][issue] += 1;
			trnModel.ndys[m][stance][sentiment] += 1;
			trnModel.ndy[m][stance] += 1;
			trnModel.nd[m] += 1;
			
			int [] results = new int [3]; 
			
			results[0] = issue;
			results[1] = sentiment;
			results[2] = stance;
			
	 		return results;
		}
		
		public void computeTheta(){
			for (int m = 0; m < trnModel.M; m++){
				for (int s = 0; s < trnModel.Y; s++) {
					for (int e = 0; e < trnModel.I; e++){
						trnModel.theta_i[m][s][e] = (trnModel.ndyi[m][s][e] + trnModel.alpha_yi[s][e]) / (trnModel.ndy[m][s] + trnModel.alphaSum_yi[s]);
					}
				}
				
			}
			for (int m = 0; m < trnModel.M; m++){
				for (int s = 0; s < trnModel.Y; s++) {
					for (int t = 0; t < trnModel.S; t++){
						trnModel.theta_s[m][s][t] = (trnModel.ndys[m][s][t] + trnModel.alpha_ys[s][t]) / (trnModel.ndy[m][s] + trnModel.alphaSum_ys[s]);
					}
				}
				
			}
			
			for (int m = 0; m < trnModel.M; m++){
					for (int s = 0; s < trnModel.Y; s++) {
							trnModel.theta_y[m][s] = (trnModel.ndy[m][s] + trnModel.alpha_y) / (trnModel.nd[m] + trnModel.alpha_y * trnModel.Y);
						}
			}
					
			
		}
		
		public void computePhi(){
			for (int e = 0; e < trnModel.I; e++){
				for (int w = 0; w < trnModel.V; w++){
					trnModel.phi_i[e][w] = (trnModel.nwi[w][e] + trnModel.beta_iw[e][w]) / (trnModel.ni[e] + trnModel.beta_isum[e]);
				}
			}
			
			for (int t = 0; t < trnModel.S; t++){
				for (int w = 0; w < trnModel.V; w++){
					trnModel.phi_s[t][w] = (trnModel.nws[w][t] + trnModel.beta_sw[t][w]) / (trnModel.ns[t] + trnModel.beta_ssum[t]);
				}
			}
			
			for (int s = 0; s < trnModel.Y; s++){
				for (int w = 0; w < trnModel.V; w++){
					trnModel.phi_y[s][w] = (trnModel.nwy[w][s] + trnModel.beta_yw[s][w]) / (trnModel.ny[s] + trnModel.beta_ysum[s]);
				}
			}
		}
		
		

}
