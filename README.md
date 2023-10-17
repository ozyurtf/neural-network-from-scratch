## Binary Cross Entropy Equations

<details>
<summary>Forward Propagation</summary>

$a_0 = x$<br>

$s_1 = W_1.a_0 + b_1$<br>

$a_1 = g_1(s_1)$<br>  

$s_2 = W_2.a_1 + b_2$<br>  

$a_2 = g_2(s_2)$<br>

$\widetilde{y} = a_2$<br>

$g_i(x) = \tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}$<br>

$g_i(x) = \sigma(x) = \frac{1}{1 + e^{-x}}$<br>

$g_i(x) = ReLU(x) = max(0, x)$<br>

$g_i(x) = Identity(x) = x$<br>

</details>

<details>
<summary>Gradients</summary><br>
        
$\frac{\partial \widetilde{y}}{\partial a_2}$<br>

$\frac{\partial a_2}{\partial s_2}$<br>

$\frac{\partial s_2}{\partial W_2}$<br>

$\frac{\partial s_2}{\partial a_1}$<br>

$\frac{\partial a_2}{\partial s_1}$<br>

$\frac{\partial s_1}{\partial W_1}$<br>

$\frac{\partial s_1}{\partial b_1}$<br>

$\frac{\partial s_2}{\partial W_2}$<br>

$\delta_1 = \frac{\partial C}{\partial \widetilde{y}}\frac{\partial \widetilde{y}}{s_2}$<br>

$\delta_2 = \frac{\partial C}{\partial \widetilde{y}}\frac{\partial \widetilde{y}}{s_2}\frac{\partial s_2}{\partial a_1}\frac{\partial a_1}{\partial s_1} = \delta_1W_2\frac{\partial a_1}{\partial s_1}$<br>

$\frac{\partial C}{\partial W_{2}}  = \delta_1\frac{\partial s_2}{\partial W_2} = \delta_1^Ta_1^T, \ a_1 = g_1(s_1)$<br>

$\frac{\partial C}{\partial W_{2}}  = \delta_1^Tg_1(s_1)^T$<br>

$\frac{\partial C}{\partial b_2} = \delta_1\frac{\partial s_2}{\partial b_2} = \delta_1^T$<br>

$\frac{\partial C}{\partial W_{1}} = \delta_2\frac{\partial s_1}{\partial W_{1}} = \delta_2^Ta_0^T$<br>

$\frac{\partial C}{\partial b_2} = \delta_2\frac{\partial s_1}{\partial b_1} = \delta_2^T$<br>

</details>

</details>

<details>
<summary>Deriving Loss Function</summary><br>

</details>

<details>
<summary>Back Propagation</summary><br>

$W_2 = W_2 - \alpha\frac{\partial C}{\partial W_{2}}$<br>

$W_1 = W_1 - \alpha\frac{\partial C}{\partial W_{1}}$<br>

$b_2 = b_2 - \alpha\frac{\partial C}{\partial b_{2}}$<br>

$b_1 = b_1 - \alpha\frac{\partial C}{\partial b_{1}}$<br>


