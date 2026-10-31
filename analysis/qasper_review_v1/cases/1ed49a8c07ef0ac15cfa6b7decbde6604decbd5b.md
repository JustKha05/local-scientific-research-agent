# qasper:1ed49a8c07ef0ac15cfa6b7decbde6604decbd5b

**Question:** What dataset does this approach achieve state of the art results on?

Paper: `qasper:1906.07701` — Distilling Translations with Visual Awareness

Groups: selection_loss, regression_complete. Human verification: **PENDING**. Everything labelled automatic below is a deterministic structural observation, not a semantic cause.

## Eligible gold evidence alternatives

QASPER completion needs any ONE eligible alternative. Optional gold from another alternative is not a failure. Normalized duplicate text in the same paper is equivalent; each hit still consumes a rank.

### Annotation `045aae4280bbf780dd9bd2c5a466c5edbd4fed80`

Evidence: [qasper:1906.07701:s5:p0](#e-qasper-1906.07701-s5-p0), [qasper:1906.07701:s5:p1](#e-qasper-1906.07701-s5-p1)

**`qasper:1906.07701:s5:p0`**

Table TABREF14 shows the results of our main experiments on the 2016 and 2018 test sets for French and German. We use Meteor BIBREF31 as the main metric, as in the WMT tasks BIBREF25 . We compare our transformer baseline to transformer models enriched with image information, as well as to the deliberation models, with or without image information.

**`qasper:1906.07701:s5:p1`**

We first note that our multimodal models achieve the state of the art performance for transformer networks (constrained models) on the English-German dataset, as compared to BIBREF30 . Second, our deliberation models lead to significant improvements over this baseline across test sets (average INLINEFORM0 , INLINEFORM1 ).

## Automatic observations (not human labels)

```json
{
  "alternatives": [
    {
      "annotation_id": "045aae4280bbf780dd9bd2c5a466c5edbd4fed80",
      "evidence_ids": [
        "qasper:1906.07701:s5:p0",
        "qasper:1906.07701:s5:p1"
      ],
      "missing_ids": [
        "qasper:1906.07701:s5:p0"
      ],
      "one_hop_available": true
    }
  ],
  "missing_gold_adjacency": [
    {
      "missing_gold": "qasper:1906.07701:s5:p0",
      "candidate_id": "qasper:1906.07701:s5:p0",
      "anchor": "qasper:1906.07701:s5:p1"
    }
  ],
  "incomplete_annotation_one_hop_available": true,
  "exact_duplicate_pairs": [],
  "high_overlap_pairs": []
}
```

An adjacent missed paragraph indicates a potential context signal, not that all neighboring text is relevant. One-hop availability uses diagnosis gold, allows additions without replacements and ignores a token budget; it is not a fixed-budget performance estimate or experiment.

## reranker top10

Metrics: `{"complete_evidence_hit@1": 0, "complete_evidence_hit@10": 0, "complete_evidence_hit@5": 0, "mrr@10": 1.0, "recall@1": 0.5, "recall@10": 0.5, "recall@5": 0.5, "status": "partial"}`

| Rank | Evidence (full text below) | Saved score | Eligible-gold equivalent | Text preview |
|---:|---|---:|---|---|
| 1 | [qasper:1906.07701:s5:p1](#e-qasper-1906.07701-s5-p1) | 5.4402995 | True | We first note that our multimodal models achieve the state of the art performance for transformer networks (constrained models) on the English-German dataset, as compared to BIBREF… |
| 2 | [qasper:1906.07701:abstract](#e-qasper-1906.07701-abstract) | 2.193753 | False | Previous work on multimodal machine translation has shown that visual information is only needed in very specific cases, for example in the presence of ambiguous words where the te… |
| 3 | [qasper:1906.07701:s7:p0](#e-qasper-1906.07701-s7-p0) | 2.0953424 | False | We have proposed a novel approach to multimodal machine translation which makes better use of context, both textual and visual. Our results show that further exploring textual cont… |
| 4 | [qasper:1906.07701:s1:p0](#e-qasper-1906.07701-s1-p0) | -6.7737756 | False | In addition to using the Multi30K dataset as is (standard setup), we probe the ability of our models to address the three linguistic phenomena where additional context has been pro… |
| 5 | [qasper:1906.07701:s4:p0](#e-qasper-1906.07701-s4-p0) | -6.7762647 | False | In this section we present results of our experiments, first in the original dataset without any source degradation (Section SECREF18 ) and then in the setup with various source de… |
| 6 | [qasper:1906.07701:s1:p2](#e-qasper-1906.07701-s1-p2) | -6.921227 | False | In this setting (AMB), we rely on the MLT dataset BIBREF11 which provides a list of source words with multiple translations in the Multi30k training set. We replace ambiguous words… |
| 7 | [qasper:1906.07701:s5:p4](#e-qasper-1906.07701-s5-p4) | -8.2346563 | False | The annotators were asked to rank randomly selected test samples according to how well they convey the meaning of the source, given the image (50 samples per language pair per anno… |
| 8 | [qasper:1906.07701:s1:p4](#e-qasper-1906.07701-s1-p4) | -8.6086006 | False | The statistics of the resulting datasets for the three degradation strategies are shown in Table TABREF10 . We note that RND and PERS are the same for language pairs as the degrada… |
| 9 | [qasper:1906.07701:s5:p5](#e-qasper-1906.07701-s5-p5) | -8.7994719 | False | Table TABREF19 shows the human evaluation results. They are consistent with the automatic evaluation results when it comes to the preference of humans towards the deliberation-base… |
| 10 | [qasper:1906.07701:s1:p3](#e-qasper-1906.07701-s1-p3) | -8.8077669 | False | In this setting (PERS), we use the Flickr Entities dataset BIBREF28 to identify all the words that were annotated by humans as corresponding to the category person. We then replace… |

## rrf_pool50 top10

Metrics: `{"complete_evidence_hit@1": 0, "complete_evidence_hit@10": 1, "complete_evidence_hit@5": 0, "mrr@10": 0.25, "recall@1": 0.0, "recall@10": 1.0, "recall@5": 0.5, "status": "complete"}`

| Rank | Evidence (full text below) | Saved score | Eligible-gold equivalent | Text preview |
|---:|---|---:|---|---|
| 1 | [qasper:1906.07701:s1:p2](#e-qasper-1906.07701-s1-p2) | 0.031754032 | False | In this setting (AMB), we rely on the MLT dataset BIBREF11 which provides a list of source words with multiple translations in the Multi30k training set. We replace ambiguous words… |
| 2 | [qasper:1906.07701:s4:p0](#e-qasper-1906.07701-s4-p0) | 0.031257631 | False | In this section we present results of our experiments, first in the original dataset without any source degradation (Section SECREF18 ) and then in the setup with various source de… |
| 3 | [qasper:1906.07701:s1:p3](#e-qasper-1906.07701-s1-p3) | 0.030309989 | False | In this setting (PERS), we use the Flickr Entities dataset BIBREF28 to identify all the words that were annotated by humans as corresponding to the category person. We then replace… |
| 4 | [qasper:1906.07701:s5:p1](#e-qasper-1906.07701-s5-p1) | 0.030282332 | True | We first note that our multimodal models achieve the state of the art performance for transformer networks (constrained models) on the English-German dataset, as compared to BIBREF… |
| 5 | [qasper:1906.07701:s0:p0](#e-qasper-1906.07701-s0-p0) | 0.030117754 | False | We build and test our MMT models on the Multi30K dataset BIBREF21 . Each image in Multi30K contains one English (EN) description taken from Flickr30K BIBREF22 and human translation… |
| 6 | [qasper:1906.07701:s7:p0](#e-qasper-1906.07701-s7-p0) | 0.029030911 | False | We have proposed a novel approach to multimodal machine translation which makes better use of context, both textual and visual. Our results show that further exploring textual cont… |
| 7 | [qasper:1906.07701:s5:p4](#e-qasper-1906.07701-s5-p4) | 0.028850145 | False | The annotators were asked to rank randomly selected test samples according to how well they convey the meaning of the source, given the image (50 samples per language pair per anno… |
| 8 | [qasper:1906.07701:s5:p5](#e-qasper-1906.07701-s5-p5) | 0.028790389 | False | Table TABREF19 shows the human evaluation results. They are consistent with the automatic evaluation results when it comes to the preference of humans towards the deliberation-base… |
| 9 | [qasper:1906.07701:s1:p0](#e-qasper-1906.07701-s1-p0) | 0.028778468 | False | In addition to using the Multi30K dataset as is (standard setup), we probe the ability of our models to address the three linguistic phenomena where additional context has been pro… |
| 10 | [qasper:1906.07701:s5:p0](#e-qasper-1906.07701-s5-p0) | 0.028370221 | True | Table TABREF14 shows the results of our main experiments on the 2016 and 2018 test sets for French and German. We use Meteor BIBREF31 as the main metric, as in the WMT tasks BIBREF… |

## Full candidate pool and related evidence

All 36 original union candidates are shown, not only heuristic matches. Gold/neighbor flags are diagnosis-only audit hints. Section titles and abstracts are not treated as adjacent paragraphs. BM25/dense scores are original source scores, not cross-encoder logits. Unstored cross-encoder scores outside top10 are **unknown**, never filled with zero.

<a id="e-qasper-1906.07701-abstract"></a>

### `qasper:1906.07701:abstract`

Section: Abstract | unit: abstract | section_index=None, paragraph_index=None

Flags: reranker_top10

Source rank/score: `{"bm25": {"rank": 2, "score": 10.096278137533016}, "dense": {"rank": 27, "score": 0.1578139364719391}}`

Saved reranker score: 2.1937530040740967

Previous work on multimodal machine translation has shown that visual information is only needed in very specific cases, for example in the presence of ambiguous words where the textual context is not sufficient. As a consequence, models tend to learn to ignore this information. We propose a translate-and-refine approach to this problem where images are only used by a second stage decoder. This approach is trained jointly to generate a good first draft translation and to improve over this draft by (i) making better use of the target language textual context (both left and right-side contexts) and (ii) making use of visual context. This approach leads to the state of the art results. Additionally, we show that it has the ability to recover from erroneous or missing words in the source language.

<a id="e-qasper-1906.07701-s0-p0"></a>

### `qasper:1906.07701:s0:p0`

Section: Data | unit: paragraph | section_index=0, paragraph_index=0

Flags: rrf_top10

Source rank/score: `{"bm25": {"rank": 9, "score": 3.990744216167837}, "dense": {"rank": 4, "score": 0.3259921669960022}}`

Saved reranker score: UNKNOWN (outside stored top10)

We build and test our MMT models on the Multi30K dataset BIBREF21 . Each image in Multi30K contains one English (EN) description taken from Flickr30K BIBREF22 and human translations into German (DE), French (FR) and Czech BIBREF23 , BIBREF24 , BIBREF25 . The dataset contains 29,000 instances for training, 1,014 for development, and 1,000 for test. We only experiment with German and French, which are languages for which we have in-house expertise for the type of analysis we present. In addition to the official Multi30K test set (test 2016), we also use the test set from the latest WMT evaluation competition, test 2018 BIBREF25 .

<a id="e-qasper-1906.07701-s0-title"></a>

### `qasper:1906.07701:s0:title`

Section: Data | unit: section_title | section_index=0, paragraph_index=None

Flags: none

Source rank/score: `{"bm25": {"rank": 29, "score": 0.0}, "dense": {"rank": 1, "score": 0.44343534111976624}}`

Saved reranker score: UNKNOWN (outside stored top10)

Data

<a id="e-qasper-1906.07701-s1-p0"></a>

### `qasper:1906.07701:s1:p0`

Section: Degradation of source | unit: paragraph | section_index=1, paragraph_index=0

Flags: reranker_top10, rrf_top10

Source rank/score: `{"bm25": {"rank": 10, "score": 3.8494765112667855}, "dense": {"rank": 9, "score": 0.28094595670700073}}`

Saved reranker score: -6.773775577545166

In addition to using the Multi30K dataset as is (standard setup), we probe the ability of our models to address the three linguistic phenomena where additional context has been proved important (Section ): ambiguities, gender-neutral words and noisy input. In a controlled experiment where we aim to remove the influence of frequency biases, we degrade the source sentences by masking words through three strategies to replace words by a placeholder: random source words, ambiguous source words and gender unmarked source words. The procedure is applied to the train, validation and test sets. For the resulting dataset generated for each setting, we compare models having access to text-only context versus additional text and multimodal contexts. We seek to get insights into the contribution of each type of context to address each type of degradation.

<a id="e-qasper-1906.07701-s1-p1"></a>

### `qasper:1906.07701:s1:p1`

Section: Degradation of source | unit: paragraph | section_index=1, paragraph_index=1

Flags: none

Source rank/score: `{"bm25": {"rank": 19, "score": 2.4924359188783964}, "dense": {"rank": 22, "score": 0.1787891387939453}}`

Saved reranker score: UNKNOWN (outside stored top10)

In this setting (RND) we simulate erroneous source words by randomly dropping source content words. We first tag the entire source sentences using the spacy toolkit BIBREF26 and then drop nouns, verbs, adjectives and adverbs and replace these with a default BLANK token. By focusing on content words, we differ from previous work that suggests that neural machine translation is robust to non-content word noise in the source BIBREF27 .

<a id="e-qasper-1906.07701-s1-p2"></a>

### `qasper:1906.07701:s1:p2`

Section: Degradation of source | unit: paragraph | section_index=1, paragraph_index=2

Flags: reranker_top10, rrf_top10

Source rank/score: `{"bm25": {"rank": 4, "score": 6.6249715759088055}, "dense": {"rank": 2, "score": 0.3759368062019348}}`

Saved reranker score: -6.921226978302002

In this setting (AMB), we rely on the MLT dataset BIBREF11 which provides a list of source words with multiple translations in the Multi30k training set. We replace ambiguous words with the BLANK token in the source language, which results in two language-specific datasets.

<a id="e-qasper-1906.07701-s1-p3"></a>

### `qasper:1906.07701:s1:p3`

Section: Degradation of source | unit: paragraph | section_index=1, paragraph_index=3

Flags: reranker_top10, rrf_top10

Source rank/score: `{"bm25": {"rank": 7, "score": 4.087996176851108}, "dense": {"rank": 5, "score": 0.3075141906738281}}`

Saved reranker score: -8.807766914367676

In this setting (PERS), we use the Flickr Entities dataset BIBREF28 to identify all the words that were annotated by humans as corresponding to the category person. We then replace such source words with the BLANK token.

<a id="e-qasper-1906.07701-s1-p4"></a>

### `qasper:1906.07701:s1:p4`

Section: Degradation of source | unit: paragraph | section_index=1, paragraph_index=4

Flags: reranker_top10

Source rank/score: `{"bm25": {"rank": 13, "score": 3.3880057692129224}, "dense": {"rank": 17, "score": 0.2171103060245514}}`

Saved reranker score: -8.608600616455078

The statistics of the resulting datasets for the three degradation strategies are shown in Table TABREF10 . We note that RND and PERS are the same for language pairs as the degradation only depends on the source side, while for AMB the words replaced depend on the target language.

<a id="e-qasper-1906.07701-s1-title"></a>

### `qasper:1906.07701:s1:title`

Section: Degradation of source | unit: section_title | section_index=1, paragraph_index=None

Flags: none

Source rank/score: `{"bm25": {"rank": 26, "score": 1.1984708800007617}, "dense": {"rank": 34, "score": 0.03863486647605896}}`

Saved reranker score: UNKNOWN (outside stored top10)

Degradation of source

<a id="e-qasper-1906.07701-s2-p0"></a>

### `qasper:1906.07701:s2:p0`

Section: Models | unit: paragraph | section_index=2, paragraph_index=0

Flags: none

Source rank/score: `{"bm25": {"rank": 25, "score": 1.4606760881489773}, "dense": {"rank": 8, "score": 0.30130109190940857}}`

Saved reranker score: UNKNOWN (outside stored top10)

Based on the models described in Section we experiment with eight variants: (a) baseline transformer model (base); (b) base with AIC (base+sum); (c) base with AIF using spacial (base+att) or object based (base+obj) image features; (d) standard deliberation model (del); (e) deliberation models enriched with image information: del+sum, del+att and del+obj.

<a id="e-qasper-1906.07701-s2-title"></a>

### `qasper:1906.07701:s2:title`

Section: Models | unit: section_title | section_index=2, paragraph_index=None

Flags: none

Source rank/score: `{"bm25": {"rank": 30, "score": 0.0}, "dense": {"rank": 6, "score": 0.3063237965106964}}`

Saved reranker score: UNKNOWN (outside stored top10)

Models

<a id="e-qasper-1906.07701-s3-p0"></a>

### `qasper:1906.07701:s3:p0`

Section: Training | unit: paragraph | section_index=3, paragraph_index=0

Flags: none

Source rank/score: `{"bm25": {"rank": 22, "score": 2.2826470756022403}, "dense": {"rank": 18, "score": 0.19712303578853607}}`

Saved reranker score: UNKNOWN (outside stored top10)

In all cases, we optimise our models with cross entropy loss. For deliberation network models, we first train the standard transformer model until convergence, and use it to initialise the encoder and first-pass decoder. For each of the training samples, we follow BIBREF19 and obtain a set of 10-best samples from the first pass decoder, with a beam search of size 10. We use these as the first-pass decoder samples. We use Adam as optimiser BIBREF29 and train the model until convergence.

<a id="e-qasper-1906.07701-s3-title"></a>

### `qasper:1906.07701:s3:title`

Section: Training | unit: section_title | section_index=3, paragraph_index=None

Flags: none

Source rank/score: `{"bm25": {"rank": 31, "score": 0.0}, "dense": {"rank": 15, "score": 0.24473321437835693}}`

Saved reranker score: UNKNOWN (outside stored top10)

Training

<a id="e-qasper-1906.07701-s4-p0"></a>

### `qasper:1906.07701:s4:p0`

Section: Results | unit: paragraph | section_index=4, paragraph_index=0

Flags: reranker_top10, rrf_top10

Source rank/score: `{"bm25": {"rank": 5, "score": 6.178195400989674}, "dense": {"rank": 3, "score": 0.34925705194473267}}`

Saved reranker score: -6.776264667510986

In this section we present results of our experiments, first in the original dataset without any source degradation (Section SECREF18 ) and then in the setup with various source degradation strategies (Section SECREF25 ).

<a id="e-qasper-1906.07701-s4-title"></a>

### `qasper:1906.07701:s4:title`

Section: Results | unit: section_title | section_index=4, paragraph_index=None

Flags: none

Source rank/score: `{"bm25": {"rank": 24, "score": 1.9041937571218708}, "dense": {"rank": 7, "score": 0.3061126172542572}}`

Saved reranker score: UNKNOWN (outside stored top10)

Results

<a id="e-qasper-1906.07701-s5-p0"></a>

### `qasper:1906.07701:s5:p0`

Section: Standard setup | unit: paragraph | section_index=5, paragraph_index=0

Flags: eligible_gold_equivalent, neighbor_of_gold, rrf_top10

Source rank/score: `{"bm25": {"rank": 11, "score": 3.7979044716490895}, "dense": {"rank": 10, "score": 0.27798253297805786}}`

Saved reranker score: UNKNOWN (outside stored top10)

Table TABREF14 shows the results of our main experiments on the 2016 and 2018 test sets for French and German. We use Meteor BIBREF31 as the main metric, as in the WMT tasks BIBREF25 . We compare our transformer baseline to transformer models enriched with image information, as well as to the deliberation models, with or without image information.

<a id="e-qasper-1906.07701-s5-p1"></a>

### `qasper:1906.07701:s5:p1`

Section: Standard setup | unit: paragraph | section_index=5, paragraph_index=1

Flags: eligible_gold_equivalent, neighbor_of_gold, reranker_top10, rrf_top10

Source rank/score: `{"bm25": {"rank": 1, "score": 13.120675475856466}, "dense": {"rank": 12, "score": 0.2662885785102844}}`

Saved reranker score: 5.4402995109558105

We first note that our multimodal models achieve the state of the art performance for transformer networks (constrained models) on the English-German dataset, as compared to BIBREF30 . Second, our deliberation models lead to significant improvements over this baseline across test sets (average INLINEFORM0 , INLINEFORM1 ).

<a id="e-qasper-1906.07701-s5-p2"></a>

### `qasper:1906.07701:s5:p2`

Section: Standard setup | unit: paragraph | section_index=5, paragraph_index=2

Flags: neighbor_of_gold

Source rank/score: `{"bm25": {"rank": 18, "score": 2.809265171707517}, "dense": {"rank": 24, "score": 0.1674354076385498}}`

Saved reranker score: UNKNOWN (outside stored top10)

Transformer-based models enriched with image information (base+sum, base+att and base+obj), on the other hand, show no major improvements with respect to the base performance. This is also the case for deliberation models with image information (del+sum, del+att, del+obj), which do not show significant improvement over the vanilla deliberation performance (del).

<a id="e-qasper-1906.07701-s5-p3"></a>

### `qasper:1906.07701:s5:p3`

Section: Standard setup | unit: paragraph | section_index=5, paragraph_index=3

Flags: none

Source rank/score: `{"bm25": {"rank": 15, "score": 3.086344246949099}, "dense": {"rank": 28, "score": 0.15653328597545624}}`

Saved reranker score: UNKNOWN (outside stored top10)

However, as it has been shown in the WMT shared tasks on MMT BIBREF23 , BIBREF24 , BIBREF25 , automatic metrics often fail to capture nuances in translation quality, such as, the ones we expect the visual modality to help with, which – according to human perception – lead to better translations. To test this assumption in our settings, we performed human evaluation involving professional translators and native speakers of both French and German (three annotators).

<a id="e-qasper-1906.07701-s5-p4"></a>

### `qasper:1906.07701:s5:p4`

Section: Standard setup | unit: paragraph | section_index=5, paragraph_index=4

Flags: reranker_top10, rrf_top10

Source rank/score: `{"bm25": {"rank": 6, "score": 5.067365226125902}, "dense": {"rank": 13, "score": 0.26069343090057373}}`

Saved reranker score: -8.23465633392334

The annotators were asked to rank randomly selected test samples according to how well they convey the meaning of the source, given the image (50 samples per language pair per annotator). For each source segment, the annotator was shown the outputs of three systems: base+att, the current MMT state-of-the-art BIBREF30 , del and del+obj. A rank could be assigned from 1 to 3, allowing ties BIBREF32 . Annotators could assign zero rank to all translations if they were judged incomprehensible. Following the common practice in WMT BIBREF32 , each system was then assigned a score which reflects the proportion of times it was judged to be better or equal other systems.

<a id="e-qasper-1906.07701-s5-p5"></a>

### `qasper:1906.07701:s5:p5`

Section: Standard setup | unit: paragraph | section_index=5, paragraph_index=5

Flags: reranker_top10, rrf_top10

Source rank/score: `{"bm25": {"rank": 8, "score": 4.0033620612091525}, "dense": {"rank": 11, "score": 0.2771734595298767}}`

Saved reranker score: -8.799471855163574

Table TABREF19 shows the human evaluation results. They are consistent with the automatic evaluation results when it comes to the preference of humans towards the deliberation-based setups, but show a more positive outlook regarding the addition of visual information (del+obj over del) for French.

<a id="e-qasper-1906.07701-s5-p6"></a>

### `qasper:1906.07701:s5:p6`

Section: Standard setup | unit: paragraph | section_index=5, paragraph_index=6

Flags: none

Source rank/score: `{"bm25": {"rank": 20, "score": 2.4304572548407615}, "dense": {"rank": 35, "score": 0.029707949608564377}}`

Saved reranker score: UNKNOWN (outside stored top10)

Manual inspection of translations suggests that deliberation setups tend to improve both the grammaticality and adequacy of the first pass outputs. For German, the most common modifications performed by the second-pass decoder are substitutions of adjectives and verbs (for test 2016, 15% and 12% respectively, of all the edit distance operations). Changes to adjectives are mainly grammatical, changes to verbs are contextual (e.g., changing laufen to rennen, both verbs mean run, but the second refers to running very fast). For French, 15% of all the changes are substitutions of nouns (for test 2016). These are again very contextual. For example, the French word travailleur (worker) is replaced by ouvrier (manual worker) in the contexts where tools, machinery or buildings are mentioned. For our analysis we used again spacy.

<a id="e-qasper-1906.07701-s5-p7"></a>

### `qasper:1906.07701:s5:p7`

Section: Standard setup | unit: paragraph | section_index=5, paragraph_index=7

Flags: none

Source rank/score: `{"bm25": {"rank": 17, "score": 2.864481425050261}, "dense": {"rank": 25, "score": 0.1668817102909088}}`

Saved reranker score: UNKNOWN (outside stored top10)

The information on detected objects is particularly helpful for specific adequacy issues. Figure FIGREF15 demonstrates some such cases. In the first case, the base+att model misses the translation of race car: the German word Rennen translates only the word race. del introduces the word car (Auto) into the translation. Finally, del+obj correctly translates the expression race car (Rennwagen) by exploiting the object information. For French, del translates the source part in a body of water, missing from the base+att translation. del+obj additionally translated the word paddling according to the detected object Paddle.

<a id="e-qasper-1906.07701-s5-title"></a>

### `qasper:1906.07701:s5:title`

Section: Standard setup | unit: section_title | section_index=5, paragraph_index=None

Flags: none

Source rank/score: `{"bm25": {"rank": 32, "score": 0.0}, "dense": {"rank": 29, "score": 0.1548779308795929}}`

Saved reranker score: UNKNOWN (outside stored top10)

Standard setup

<a id="e-qasper-1906.07701-s6-p0"></a>

### `qasper:1906.07701:s6:p0`

Section: Source degradation setup | unit: paragraph | section_index=6, paragraph_index=0

Flags: none

Source rank/score: `{"bm25": {"rank": 12, "score": 3.5889799963875895}, "dense": {"rank": 14, "score": 0.2547391951084137}}`

Saved reranker score: UNKNOWN (outside stored top10)

Results of our source degradation experiments are shown in Table TABREF20 . A first observation is that – as with the standard setup – the performance of our deliberation models is overall better than that of the base models. The results of the multimodal models differ for German and French. For German, del+obj is the most successful configuration and shows statistically significant improvements over base for all setups. Moreover, for RND and AMB, it shows statistically significant improvements over del. However, especially for RND and AMB, del and del+sum are either the same or slightly worse than base.

<a id="e-qasper-1906.07701-s6-p1"></a>

### `qasper:1906.07701:s6:p1`

Section: Source degradation setup | unit: paragraph | section_index=6, paragraph_index=1

Flags: none

Source rank/score: `{"bm25": {"rank": 28, "score": 1.151895618524432}, "dense": {"rank": 19, "score": 0.19041571021080017}}`

Saved reranker score: UNKNOWN (outside stored top10)

For French, all the deliberation models show statistically significant improvements over base (average INLINEFORM0 , INLINEFORM1 ), but the image information added to del only improve scores significantly for test 2018 RND.

<a id="e-qasper-1906.07701-s6-p2"></a>

### `qasper:1906.07701:s6:p2`

Section: Source degradation setup | unit: paragraph | section_index=6, paragraph_index=2

Flags: none

Source rank/score: `{"bm25": {"rank": 14, "score": 3.316199484609198}, "dense": {"rank": 36, "score": -0.011356193572282791}}`

Saved reranker score: UNKNOWN (outside stored top10)

This difference in performances for French and German is potentially related to the need of more significant restructurings while translating from English into German. This is where a more complex del+obj architecture is more helpful. This is especially true for RND and AMB setups where blanked words could also be verbs, the part-of-speech most influenced by word order differences between English and German (see the decreasing complexity of translations for del and del+obj for the example (c) in Figure FIGREF21 ).

<a id="e-qasper-1906.07701-s6-p3"></a>

### `qasper:1906.07701:s6:p3`

Section: Source degradation setup | unit: paragraph | section_index=6, paragraph_index=3

Flags: none

Source rank/score: `{"bm25": {"rank": 21, "score": 2.4043426652637576}, "dense": {"rank": 21, "score": 0.1878352165222168}}`

Saved reranker score: UNKNOWN (outside stored top10)

To get an insight into the contribution of different contexts to the resolution of blanks, we performed manual analysis of examples coming from the English-German base, del and del+obj setups (50 random examples per setup), where we count correctly translated blanks per system.

<a id="e-qasper-1906.07701-s6-p4"></a>

### `qasper:1906.07701:s6:p4`

Section: Source degradation setup | unit: paragraph | section_index=6, paragraph_index=4

Flags: none

Source rank/score: `{"bm25": {"rank": 16, "score": 3.033886016579836}, "dense": {"rank": 23, "score": 0.1769314408302307}}`

Saved reranker score: UNKNOWN (outside stored top10)

The results are shown in Table TABREF27 . As expected, they show that the RND and AMB blanks are more difficult to resolve (at most 40% resolved as compared to 61% for PERS). Translations of the majority of those blanks tend to be guessed by the textual context alone (especially for verbs). Image information is more helpful for PERS: we observe an increase of 10% in resolved blanks for del+obj as compared to del. However, for PERS the textual context is still enough in the majority of the cases: models tend to associate men with sports or women with cooking and are usually right (see Figure FIGREF21 example (c)).

<a id="e-qasper-1906.07701-s6-p5"></a>

### `qasper:1906.07701:s6:p5`

Section: Source degradation setup | unit: paragraph | section_index=6, paragraph_index=5

Flags: none

Source rank/score: `{"bm25": {"rank": 27, "score": 1.1633767141374192}, "dense": {"rank": 33, "score": 0.07295018434524536}}`

Saved reranker score: UNKNOWN (outside stored top10)

The cases where image helps seem to be those with rather generic contexts: see Figure FIGREF21 (b) where enjoying a summer day is not associated with any particular gender and make other models choose homme (man) or femme (woman), and only base+obj chooses enfant (child) (the option closest to the reference).

<a id="e-qasper-1906.07701-s6-p6"></a>

### `qasper:1906.07701:s6:p6`

Section: Source degradation setup | unit: paragraph | section_index=6, paragraph_index=6

Flags: none

Source rank/score: `{"bm25": {"rank": 33, "score": 0.0}, "dense": {"rank": 32, "score": 0.0877174437046051}}`

Saved reranker score: UNKNOWN (outside stored top10)

In some cases detected objects are inaccurate or not precise enough to be helpful (e.g., when an object Person is detected) and can even harm correct translations.

<a id="e-qasper-1906.07701-s6-title"></a>

### `qasper:1906.07701:s6:title`

Section: Source degradation setup | unit: section_title | section_index=6, paragraph_index=None

Flags: none

Source rank/score: `{"bm25": {"rank": 34, "score": 0.0}, "dense": {"rank": 26, "score": 0.16287779808044434}}`

Saved reranker score: UNKNOWN (outside stored top10)

Source degradation setup

<a id="e-qasper-1906.07701-s7-p0"></a>

### `qasper:1906.07701:s7:p0`

Section: Conclusions | unit: paragraph | section_index=7, paragraph_index=0

Flags: reranker_top10, rrf_top10

Source rank/score: `{"bm25": {"rank": 3, "score": 9.13152707658727}, "dense": {"rank": 16, "score": 0.21978896856307983}}`

Saved reranker score: 2.0953423976898193

We have proposed a novel approach to multimodal machine translation which makes better use of context, both textual and visual. Our results show that further exploring textual context through deliberation networks already leads to better results than the previous state of the art. Adding visual information, and in particular structural representations of this information, proved beneficial when input text contains noise and the language pair requires substantial restructuring from source to target. Our findings suggest that the combination of a deliberation approach and information from additional modalities is a promising direction for machine translation that is robust to noisy input. Our code and pre-processing scripts are available at https://github.com/ImperialNLP/MMT-Delib.

<a id="e-qasper-1906.07701-s7-title"></a>

### `qasper:1906.07701:s7:title`

Section: Conclusions | unit: section_title | section_index=7, paragraph_index=None

Flags: none

Source rank/score: `{"bm25": {"rank": 35, "score": 0.0}, "dense": {"rank": 31, "score": 0.11511582136154175}}`

Saved reranker score: UNKNOWN (outside stored top10)

Conclusions

<a id="e-qasper-1906.07701-s8-p0"></a>

### `qasper:1906.07701:s8:p0`

Section: Acknowledgments | unit: paragraph | section_index=8, paragraph_index=0

Flags: none

Source rank/score: `{"bm25": {"rank": 23, "score": 2.2064195386725896}, "dense": {"rank": 20, "score": 0.18857967853546143}}`

Saved reranker score: UNKNOWN (outside stored top10)

The authors thank the anonymous reviewers for their useful feedback. This work was supported by the MultiMT (H2020 ERC Starting Grant No. 678017) and MMVC (Newton Fund Institutional Links Grant, ID 352343575) projects. We also thank the annotators for their valuable help.

<a id="e-qasper-1906.07701-s8-title"></a>

### `qasper:1906.07701:s8:title`

Section: Acknowledgments | unit: section_title | section_index=8, paragraph_index=None

Flags: none

Source rank/score: `{"bm25": {"rank": 36, "score": 0.0}, "dense": {"rank": 30, "score": 0.1292494535446167}}`

Saved reranker score: UNKNOWN (outside stored top10)

Acknowledgments

## Human verification (not completed by packet generation)

- reviewer / date:
- confirmed labels (or uncertain):
- evidence IDs and exact supporting text:
- counterevidence / alternative explanation:
- support for duplicate/context hypothesis:
