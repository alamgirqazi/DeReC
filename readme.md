<h1 align="center" style="font-weight: bold; font-size: 2.5rem;">
  When retrieval outperforms generation: <span style="font-size: 1.5rem; font-weight: normal; color: #555;">Dense evidence retrieval for
scalable fake news detection</span>
</h1>


<div align="center">

Alamgir Munir Qazi<sup>1</sup> &emsp; John P. McCrae<sup>2</sup> &emsp; Jamal Abdul Nasir<sup>1</sup>  

<sup>1</sup>School of Computer Science, University of Galway, Ireland  
<sup>2</sup>Research Ireland Insight Centre and ADAPT Centre, University of Galway, Ireland  

</div>

 <div class="content has-text-justified">
          <p>
The proliferation of misinformation necessitates robust yet computationally efficient fact verification systems. While current state-of-the-art approaches leverage Large Language Models (LLMs) for generating explanatory rationales, these methods face significant computational barriers and hallucination risks in real-world deployments. We present DeReC (Dense Retrieval Classification), a lightweight framework that demonstrates how general-purpose text embeddings can effectively replace autoregressive LLM-based approaches in fact verification tasks. By combining dense retrieval with specialized classification, our system achieves better accuracy while being significantly more efficient. DeReC outperforms explanation-generating LLMs in efficiency, reducing runtime by 95% on RAWFC (23 minutes 36 seconds vs. 454 minutes 12 seconds) and by 92% on LIAR-RAW (134 minutes 14 seconds vs. 1692 minutes 23 seconds), showcasing its effectiveness across varying dataset sizes. On the RAWFC dataset, DeReC achieves an F1 score of 65.58%, surpassing state of the art method L-Defense (61.20%). Our results demonstrate that carefully engineered retrieval-based systems can match or exceed LLM performance in specialized tasks while being significantly more practical for real-world deployment.
          </p>
        </div>


## Code 

<p><i>The code will be public once the paper is published.</i></p>
