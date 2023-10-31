## Neural Network 

<details>
<summary>Forward Propagation</summary>

$a_0 = x$<br>

$s_1 = W_1.a_0 + b_1$<br>

$a_1 = g_1(s_1)$<br>  

$s_2 = W_2.a_1 + b_2$<br>  

$a_2 = g_2(s_2)$<br>

$\widetilde{y} = a_2$<br>

$g_i(s_i) = \tanh(s_i) = \frac{e^{s_i} - e^{-s_i}}{e^{s_i} + e^{-s_i}}$<br>

$g_i(s_i) = \sigma(s_i) = \frac{1}{1 + e^{-s_i}}$<br>

$g_i(s_i) = ReLU(s_i) = max(0, s_i)$<br>

$g_i(s_i) = Identity(s_i) = s_i$<br>

</details>

<details>
<summary>Gradients</summary><br>

$\frac{\partial g_i(s_i)}{\partial s_i}  = \frac{\partial tanh(s_i)}{\partial s_i} = 1 - \tanh^2(s_i)$<br>

$\frac{\partial g_i(x)}{\partial s_i}  = \frac{\partial \sigma(s_i)}{\partial s_i} = \sigma(s_i) \cdot (1 - \sigma(s_i))$<br>

$\frac{\partial g_i(x)}{\partial s_i}  = \frac{\partial ReLU(s_i)}{\partial s_i} = M$<br> 

$\frac{\partial g_i(x)}{\partial s_i} = \frac{\partial Identity(s_i)}{\partial s_i} = I_1$<br>
        
$\frac{\partial \widetilde{y}}{\partial a_2} = I_2$<br>

$\frac{\partial a_2}{\partial s_2} = \frac{\partial g_2(s_2)}{\partial s_2}$<br>

$\frac{\partial s_2}{\partial W_2} = a_1^T = g_1(s_1)^T$<br>

$\frac{\partial s_2}{\partial b_2} = I_3$<br>

$\frac{\partial s_2}{\partial a_1} = W_2^T$<br>

$\frac{\partial a_1}{\partial s_1} = \frac{\partial g_1(s_1)}{\partial s_1}$<br>

$\frac{\partial s_1}{\partial W_1} = a_0^T = x^T$<br>

$\frac{\partial s_1}{\partial a_0} = W_1^T$<br>

$\frac{\partial s_1}{\partial b_1} = I_4$<br>

$M_{kj} = 0$, $M_{kk} = 1 \text{ if } s_{i_k} > 0$, $M_{kk} = 0 \text{ if } s_{i_k} \leq 0$<br>

</details>

</details>

<details>
<summary>Deriving Loss Function</summary><br>

$\text{BCE}(y, \hat{y}) = - \frac{1}{n} \sum_{i=1} \left( y \cdot \log(\hat{y}) + (1 - y) \cdot \log(1 - \hat{y}) \right)$

$\text{MSE}(y, \hat{y}) = \frac{1}{n} \sum_{i=1} (y_i - \hat{y}_i)^2$

</details>

<details>
<summary>Back Propagation</summary><br>

$\frac{\partial C}{\partial \widetilde{y}} = \frac{\partial \text{BCE}}{\partial \hat{y}} = -\left(\frac{y}{\hat{y}} - \frac{1 - y}{1 - \hat{y}}\right)$

$\frac{\partial C}{\partial \widetilde{y}} = \frac{\partial \text{MSE}}{\partial \hat{y}} = -\frac{2}{n} \sum_{i=1} (y_i - \hat{y}_i)$


$\delta_1 = \frac{\partial C}{\partial \widetilde{y}}\frac{\partial \widetilde{y}}{s_2}$<br>

$\delta_2 = \frac{\partial C}{\partial \widetilde{y}}\frac{\partial \widetilde{y}}{s_2}\frac{\partial s_2}{\partial a_1}\frac{\partial a_1}{\partial s_1} = \delta_1W_2\frac{\partial a_1}{\partial s_1}$<br>

$\frac{\partial C}{\partial W_{2}}  = \delta_1\frac{\partial s_2}{\partial W_2} = \delta_1^Ta_1^T, \ a_1 = g_1(s_1)$<br>

$\frac{\partial C}{\partial W_{2}}  = \delta_1^Tg_1(s_1)^T$<br>

$\frac{\partial C}{\partial b_2} = \delta_1\frac{\partial s_2}{\partial b_2} = \delta_1^T$<br>

$\frac{\partial C}{\partial W_{1}} = \delta_2\frac{\partial s_1}{\partial W_{1}} = \delta_2^Ta_0^T$<br>

$\frac{\partial C}{\partial b_2} = \delta_2\frac{\partial s_1}{\partial b_1} = \delta_2^T$<br>

$W_2 = W_2 - \alpha\frac{\partial C}{\partial W_{2}}$<br>

$W_1 = W_1 - \alpha\frac{\partial C}{\partial W_{1}}$<br>

$b_2 = b_2 - \alpha\frac{\partial C}{\partial b_{2}}$<br>

$b_1 = b_1 - \alpha\frac{\partial C}{\partial b_{1}}$<br>


