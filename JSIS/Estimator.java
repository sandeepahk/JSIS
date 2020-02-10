package jsis;

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
		 * @return topic id
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
			trnModel.ndi[m][issue] -= 1;
			trnModel.nds[m][sentiment] -= 1;
			trnModel.ndis[m][issue][sentiment] -= 1;
			trnModel.ndisy[m][issue][sentiment][stance] -= 1;
			trnModel.nd[m] -= 1;
		
			
			
			double Ialpha = trnModel.alpha_i * trnModel.I;
			double Salpha = trnModel.alpha_s * trnModel.S; 
						
			
			//do multinominal sampling via cumulative method
			for (int i = 0; i < trnModel.I; i++){
				for (int s = 0; s < trnModel.S; s++) {
					for (int y = 0; y < trnModel.Y; y++) {
						trnModel.p[i][s][y] = (trnModel.nwi[w][i] + trnModel.beta_iw[i][w])/(trnModel.ni[i] + trnModel.beta_isum[i]) *
								(trnModel.nws[w][s] + trnModel.beta_sw[s][w])/(trnModel.ns[s] + trnModel.beta_ssum[s]) *
								(trnModel.nwy[w][y] + trnModel.beta_yw[y][w])/(trnModel.ny[y] + trnModel.beta_ysum[y]) *
								(trnModel.ndi[m][i] + trnModel.alpha_i)/(trnModel.nd[m] + Ialpha) *
								(trnModel.nds[m][s] + trnModel.alpha_s)/(trnModel.nd[m] + Salpha) *
								(trnModel.ndisy[m][i][s][y] + trnModel.alpha_isy[i][s][y])/(trnModel.ndis[m][i][s] + trnModel.alphaSum_is[i][s]);
					}					
						
				}				
				
			}
			
		
			// cumulate multinomial parameters
			for (int i = 0; i < trnModel.I; i++){
				for (int s = 0; s < trnModel.S; s++) {
					for (int y = 0; y < trnModel.Y; y++) {
						if(y == 0) {
							if( s == 0) {
								if(i == 0) {
										continue;
								}
								else {
									trnModel.p[i][s][y] += trnModel.p[i - 1][trnModel.S - 1][trnModel.Y - 1];
								}
								
								
							}
							else {
								trnModel.p[i][s][y] += trnModel.p[i][s - 1][trnModel.Y - 1];
							}
						}
						else {
							trnModel.p[i][s][y] += trnModel.p[i][s][y - 1];
							
							
						}
						
					}
					
						
						
				}
				
			}
			
			// scaled sample because of unnormalized p[]
			double u = Math.random() * trnModel.p[trnModel.I - 1][trnModel.S - 1][trnModel.Y - 1];
			
			boolean loopbreak = false;
			for (issue = 0; issue < trnModel.I ; issue++){
				for (sentiment = 0; sentiment < trnModel.S; sentiment++) {
					for (stance = 0; stance < trnModel.Y; stance++) {
						if (trnModel.p[issue][sentiment][stance] > u) {//sample topic w.r.t distribution p
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
			
			
			
			trnModel.nwi[w][issue] += 1;
			trnModel.nws[w][sentiment] += 1;
			trnModel.nwy[w][stance] += 1;
			trnModel.ni[issue] += 1;
			trnModel.ns[sentiment] += 1;
			trnModel.ny[stance] += 1;
			trnModel.ndi[m][issue] += 1;
			trnModel.nds[m][sentiment] += 1;
			trnModel.ndis[m][issue][sentiment] += 1;
			trnModel.ndisy[m][issue][sentiment][stance] += 1;
			trnModel.nd[m] += 1;
			
			int [] results = new int [3]; 
			
			results[0] = issue;
			results[1] = sentiment;
			results[2] = stance;
			
			
	 		return results;
		}
		
		public void computeTheta(){
			for (int m = 0; m < trnModel.M; m++){
				for (int i = 0; i < trnModel.I; i++){
					trnModel.theta_i[m][i] = (trnModel.ndi[m][i] + trnModel.alpha_i) / (trnModel.nd[m] + trnModel.I * trnModel.alpha_i);
				}
			}
			
			for (int m = 0; m < trnModel.M; m++){
				for (int s = 0; s < trnModel.S; s++){
					trnModel.theta_s[m][s] = (trnModel.nds[m][s] + trnModel.alpha_s) / (trnModel.nd[m] + trnModel.S * trnModel.alpha_s);
				}
			}
						
			for (int m = 0; m < trnModel.M; m++){
				for (int i = 0; i < trnModel.I; i++){
					for (int s = 0; s < trnModel.S; s++) {
						for (int y = 0; y < trnModel.Y; y++) {
							trnModel.theta_y[m][i][s][y] = (trnModel.ndisy[m][i][s][y] + trnModel.alpha_isy[i][s][y]) / (trnModel.ndis[m][i][s] + trnModel.alphaSum_is[i][s]);
						}
						
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
			
			for (int y = 0; y < trnModel.Y; y++){
				for (int w = 0; w < trnModel.V; w++){
					trnModel.phi_y[y][w] = (trnModel.nwy[w][y] + trnModel.beta_yw[y][w]) / (trnModel.ny[y] + trnModel.beta_ysum[y]);
				}
			}
		}
		
		
		

}
