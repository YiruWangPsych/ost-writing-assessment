# Appendix A: Variable Definitions

This appendix provides definitions for all variables analyzed in the present study.

## A.1 Writing Quality Scale (WQS-HE) Outcome Dimensions

Human raters evaluated writing samples using the Writing Quality Scale for Higher Education (Stuart & Barnett, 2024). Each dimension is scored from 1 (highest quality) to 4 (lowest quality), then reverse-coded so higher values indicate better quality. The sum of six dimensions serves as the primary outcome variable (M = 17.50, SD = 2.12, range: 10.6–21.6).

**Content and development.** Quality of ideas and their elaboration. High scores reflect fully extended and elaborated ideas using descriptive detail that engages the reader; low scores indicate list-like writing with minimal extension.

**Structure and organization.** Logical arrangement of content into paragraphs with connectives linking sentences and smooth transitions. High scores reflect clear cohesion and logical progression; low scores indicate disorganized text.

**Vocabulary.** Appropriateness and variety of word choice. High scores reflect precise, well-chosen words that accurately convey intended meaning; low scores indicate inaccurate or inappropriate word choices.

**Sentence structure.** Grammatical accuracy and sentence variety. High scores reflect well-formed, grammatically correct sentences with structural variety; low scores indicate frequent grammatical errors.

**Punctuation.** Accuracy of punctuation, capitalization, and apostrophes. High scores reflect consistent accuracy; low scores indicate frequent errors affecting readability.

**Spelling.** Accuracy of spelling. High scores reflect few or no errors; low scores indicate frequent misspellings.

## A.2 OST Linguistic Features

The Open-Source Toolkit (OST) extracts 30 linguistic features using NLTK for tokenization and lexical analysis, SpaCy for dependency parsing and clause identification, and LanguageTool for error detection.

### Basic Production Features

**Word count (OST_word_count).** Total number of alphabetic tokens excluding punctuation and numerals, indicating text length and writing fluency.

**Mean length of T-unit (OST_mean_length_t_unit).** Average words per T-unit (minimal terminable unit: one main clause plus attached subordinates). A developmental index of syntactic complexity.

**Number of T-units (OST_num_t_units).** Count of minimal terminable units identified through dependency parsing of clause boundaries.

**T-units per sentence (OST_t_units_per_sentence).** Ratio of T-units to sentences, indicating clause packaging within sentence boundaries.

**Dependent clauses per T-unit (OST_dependent_clauses_per_t_unit).** Ratio of dependent clauses to T-units, measuring syntactic subordination through embedded structures.

### Error Detection Features

**Error count (OST_error_count).** Grammatical and mechanical errors detected using LanguageTool, an open-source grammar checker with over 3,000 English rules covering spelling, grammar, punctuation, and style.

### Lexical Features

**Vocabulary sophistication (OST_vocabulary_sophistication).** Average depth within the WordNet lexical hierarchy, normalized to 0–1. Deeper synsets represent more specific, specialized vocabulary.

**Lexical diversity (OST_lexical_diversity).** Type-token ratio (unique word types / total tokens). Higher values indicate greater vocabulary range and less repetition.

**Polysemous word count (OST_polysemy_word).** Content words with multiple WordNet senses, indicating semantic flexibility.

**Context-sensitive word count (OST_context_sensitive_count).** Words whose surface form differs from the primary lemma of their first WordNet sense, capturing non-prototypical word usage.

**Word concreteness (OST_word_concreteness).** Proportion of concrete versus abstract words. Concrete words evoke mental images and are generally easier to process.

**Bigram count (OST_bigram_count).** Significant word pair collocations (frequency ≥ 2, PMI > 0), indicating meaningful word associations.

**Bigram diversity (OST_bigram_diversity).** Ratio of unique bigram collocations to total words, capturing phrasal variety.

**Word length variance (OST_word_length_variance).** Variance in character count per word, indicating heterogeneity combining short function words with longer content words.

**Syllable variance (OST_syllable_variance).** Variance in syllable count per word, capturing phonological complexity variation.

### Syntactic Features

**Sentence type diversity (OST_sentence_type_diversity).** Normalized entropy across five sentence types (declarative, interrogative, exclamatory, complex, compound), reflecting structural variety.

**Average sentence length (OST_avg_sentence_length).** Mean words per sentence. Longer sentences generally indicate greater syntactic maturity.

**Number of complex sentences (OST_num_complex_sentences).** Sentences exceeding 15 words, typically involving subordination or coordination.

**Adjective-noun pairs (OST_adj_noun_pairs).** Count of adjective-noun collocations, indicating descriptive language use and nominal modification.

**Syntactic simplicity (OST_syntactic_simplicity).** Inverse function of average sentence length, approximating Coh-Metrix PCSYNz. Higher values indicate simpler structures.

**Information density (OST_information_density).** Ratio of content words (nouns, verbs, adjectives, adverbs) to total words. Higher density indicates more semantically loaded text.

### Advanced Features

**Flesch-Kincaid grade level (OST_flesch_kincaid_grade_level).** U.S. grade level for comprehension: 0.39 × (words/sentences) + 11.8 × (syllables/words) − 15.59.

**Text ease (OST_text_ease).** Composite measure combining grade-level ease, syntactic simplicity, word concreteness, and cohesion components. Higher scores indicate more accessible text.

**Narrativity score (OST_narrativity_score).** Degree to which text resembles oral, story-like discourse with familiar content and world knowledge references.

**Referential cohesion (OST_referential_cohesion).** Mean Jaccard similarity for content word overlap between adjacent sentences, capturing local coherence.

**Deep cohesion (OST_deep_cohesion).** Density of logical connectives (causal, temporal, contrastive, additive) per word, signaling explicit logical relationships.

**Argumentative indicators (OST_argumentative_indicators).** Density of claim, evidence, reasoning, and counterargument markers.

**Argumentation completeness (OST_argumentation_completeness).** Proportion of argumentation element types present (0–1 scale).

**Argumentation balance (OST_argumentation_balance).** Evenness of distribution across argumentation element types.

**Argumentation score (OST_argumentation_score).** Composite: (completeness + balance) / 2.

---

**Note.** The OST integrates NLTK for tokenization and lexical analysis, SpaCy (en_core_web_sm) for dependency parsing and clause identification, and LanguageTool for error detection. Java Runtime Environment (version 8 or higher) is required for error detection functionality.
