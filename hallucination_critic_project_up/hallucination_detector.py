import re
from collections import defaultdict
from typing import List, Dict, Set
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class HallucinationCritic:
    """
    A Critic model that identifies hallucinations in abstractive summaries
    and provides computational traces explaining why claims are false.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))

        # ✅ GENERALIZED: Pattern-based type classification
        # Instead of hardcoded lists, we use linguistic/structural patterns
        # to decide what type a capitalized phrase belongs to.

        self.entity_type_rules = [
            # PUBLICATION: ends with known media suffixes or contains 'Daily','Times','Review','Journal','Magazine'
            ('PUBLICATION', r'(?i)\b\w+(?:\s+\w+)*\s+(?:Daily|Times|Review|Journal|Magazine|Post|Tribune|Gazette|Weekly|Report|Observer|Monitor|Chronicle|Herald)\b'),
            ('PUBLICATION', r'(?i)\b(?:Nature|Science|Cell|Lancet|Forbes|Bloomberg|Reuters|BBC|CNN|NPR)\s*\w*\b'),

            # INSTITUTION: universities, hospitals, schools, organizations, agencies
            ('INSTITUTION', r'(?i)\b\w+(?:\s+\w+)*\s+(?:University|College|Institute|School|Hospital|Clinic|Academy|Foundation|Association|Authority|Agency|Bureau|Ministry|Department|Committee|Council|Board|Centre|Center)\b'),
            ('INSTITUTION', r'(?i)\b(?:WHO|UN|IMF|NATO|WTO|UNICEF|UNESCO|FBI|CIA|NASA|FDA|CDC|NIH|MIT|UCLA|NYU)\b'),

            # LOCATION: countries, cities, regions — detected by geographic suffixes/words
            ('LOCATION', r'(?i)\b\w+(?:\s+\w+)*\s+(?:Island|Islands|Peninsula|Republic|Kingdom|Empire|County|Province|State|Prefecture|Territory|Ocean|Sea|River|Lake|Mountain|Valley|Desert|Forest|Park|City|Town|Bay|Gulf|Cape|Coast|Region)\b'),
            ('LOCATION', r'(?i)\b(?:North|South|East|West|Central|Upper|Lower|Greater|New|Old)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b'),
            ('LOCATION', r'\b(?:Africa|Europe|Asia|America|Australia|Antarctica|Arctic|Pacific|Atlantic|Mediterranean)\b'),
            # ✅ "United X" — covers United States, United Kingdom, United Arab Emirates
            ('LOCATION', r'(?i)\b(?:the\s+)?United\s+(?:States|Kingdom|Arab|Nations|Emirates)\b'),
            # ✅ Country name suffixes — e.g. Australia, Indonesia, Malaysia, Colombia, Brazil
            ('LOCATION', r'\b[A-Z][a-z]+(?:land|stan|istan|esia|nesia|frica|rica|ica|nia|lia|tia|sia|cia|via|bia|gia|mia|dia|ria|ina|ico|ile|uay|way)\b'),
            # ✅ "X of Y" geographic constructs — Gulf of Mexico, Republic of Korea
            ('LOCATION', r'\b[A-Z][a-z]+\s+of\s+[A-Z][a-z]+\b'),

            # ORG: companies, corporations
            ('ORG', r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc|Corp|LLC|Ltd|Co|Group|Holdings|Ventures|Technologies|Solutions|Industries|Enterprises)\.?\b'),

            # PERSON: typical person name pattern — 2-3 capitalized words, no org/location keywords
            ('PERSON', r'\b(?:Dr|Mr|Mrs|Ms|Prof|Sir|Dame|Lord)\.\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'),
        ]

        # Fallback patterns for anything not caught above
        self.fallback_patterns = {
            'DATE':    r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b|\b\d{1,2}/\d{1,2}/\d{4}\b',
            'MONEY':   r'\$\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:billion|million|thousand)?',
            'PERCENT': r'\d+(?:\.\d+)?%',
        }

        # Generic capitalized phrase pattern — classified after type rules
        self.capitalized_phrase_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b'

    def _classify_entity(self, phrase: str) -> str:
        """
        Generalized entity classifier.
        Tries each type rule in order; falls back to PERSON if none match.
        """
        for ent_type, pattern in self.entity_type_rules:
            if re.search(pattern, phrase):
                return ent_type
        # If no rule matches, treat as PERSON (most common fallback for named phrases)
        return 'PERSON'

    def tokenize(self, text: str) -> List[str]:
        text = re.sub(r'(?<!\d)[.,;:!?()](?!\d)', ' ', text)
        return text.split()

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Generalized entity extraction.
        Classifies every capitalized phrase using rule-based type detection.
        """
        entities = defaultdict(set)

        # Step 1: Extract all capitalized phrases
        raw_phrases = re.findall(self.capitalized_phrase_pattern, text)
        for phrase in raw_phrases:
            ent_type = self._classify_entity(phrase)
            entities[ent_type].add(phrase)

        # Step 2: Extract pattern-based types (DATE, MONEY, PERCENT)
        for ent_type, pattern in self.fallback_patterns.items():
            matches = re.findall(pattern, text)
            for m in matches:
                entities[ent_type].add(m)

        return {k: list(v) for k, v in entities.items()}

    def extract_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def extract_keywords(self, text: str, top_n: int = 10) -> Set[str]:
        try:
            sentences = self.extract_sentences(text)
            if len(sentences) < 2:
                sentences = [text, text]
            tfidf_matrix = self.vectorizer.fit_transform(sentences)
            feature_names = self.vectorizer.get_feature_names_out()
            avg_scores = np.array(tfidf_matrix.mean(axis=0)).flatten()
            top_indices = avg_scores.argsort()[-top_n:][::-1]
            return set(feature_names[i] for i in top_indices)
        except:
            words = self.tokenize(text.lower())
            word_freq = defaultdict(int)
            for word in words:
                if len(word) > 3:
                    word_freq[word] += 1
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            return set(word for word, _ in sorted_words[:top_n])

    def check_entity_hallucination(self, source_text: str, summary_text: str) -> Dict:
        source_entities = self.extract_entities(source_text)
        summary_entities = self.extract_entities(summary_text)

        hallucinated_entities = defaultdict(list)
        verified_entities = defaultdict(list)

        all_source_entities = set()
        for ent_list in source_entities.values():
            all_source_entities.update([e.lower() for e in ent_list])

        for ent_type, entities in summary_entities.items():
            for entity in entities:
                if entity.lower() not in all_source_entities:
                    hallucinated_entities[ent_type].append(entity)
                else:
                    verified_entities[ent_type].append(entity)

        return {
            'hallucinated': dict(hallucinated_entities),
            'verified': dict(verified_entities),
            'source_entities': source_entities,
            'summary_entities': summary_entities
        }

    def check_numerical_hallucination(self, source_text: str, summary_text: str) -> Dict:
        source_numbers = re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', source_text)
        summary_numbers = re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', summary_text)

        source_norm = [n.replace(',', '') for n in source_numbers]
        summary_norm = [n.replace(',', '') for n in summary_numbers]

        hallucinated_numbers, verified_numbers = [], []
        for i, num in enumerate(summary_norm):
            if num not in source_norm:
                hallucinated_numbers.append(summary_numbers[i])
            else:
                verified_numbers.append(summary_numbers[i])

        return {
            'hallucinated': hallucinated_numbers,
            'verified': verified_numbers,
            'source_numbers': source_numbers
        }

    def compute_semantic_similarity(self, source_text: str, summary_text: str) -> float:
        try:
            tfidf_matrix = self.vectorizer.fit_transform([source_text, summary_text])
            return float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
        except:
            return 0.0

    def check_keyword_coverage(self, source_text: str, summary_text: str) -> Dict:
        source_keywords = self.extract_keywords(source_text)
        summary_keywords = self.extract_keywords(summary_text)
        summary_tokens = set(self.tokenize(summary_text.lower()))

        covered = source_keywords.intersection(summary_tokens)
        missing = source_keywords - summary_tokens
        extra = summary_keywords - source_keywords
        coverage_rate = len(covered) / len(source_keywords) if source_keywords else 0

        return {
            'covered': list(covered),
            'missing': list(missing),
            'extra': list(extra),
            'coverage_rate': coverage_rate
        }

    def generate_hallucination_report(self, source_text: str, summary_text: str) -> Dict:
        entity_check    = self.check_entity_hallucination(source_text, summary_text)
        numerical_check = self.check_numerical_hallucination(source_text, summary_text)
        keyword_check   = self.check_keyword_coverage(source_text, summary_text)
        semantic_sim    = self.compute_semantic_similarity(source_text, summary_text)

        hallucination_indicators, total_checks = 0, 0

        total_summary_entities = sum(len(v) for v in entity_check['summary_entities'].values())
        if total_summary_entities > 0:
            total_hall = sum(len(v) for v in entity_check['hallucinated'].values())
            hallucination_indicators += total_hall / total_summary_entities
            total_checks += 1

        total_nums = len(numerical_check['hallucinated']) + len(numerical_check['verified'])
        if total_nums > 0:
            hallucination_indicators += len(numerical_check['hallucinated']) / total_nums
            total_checks += 1

        hallucination_indicators += (1 - keyword_check['coverage_rate'])
        total_checks += 1

        hallucination_indicators += (1 - semantic_sim)
        total_checks += 1

        hallucination_score = hallucination_indicators / total_checks if total_checks > 0 else 0
        verdict = 'HALLUCINATION DETECTED' if hallucination_score > 0.35 else 'SUMMARY VERIFIED'

        return {
            'hallucination_score': hallucination_score,
            'semantic_similarity': semantic_sim,
            'entity_analysis': entity_check,
            'numerical_analysis': numerical_check,
            'keyword_analysis': keyword_check,
            'explanation': self._generate_explanation(entity_check, numerical_check, keyword_check, semantic_sim, hallucination_score),
            'verdict': verdict,
            'computational_trace': self._generate_computational_trace(entity_check, numerical_check, keyword_check, semantic_sim, hallucination_score),
            'false_reasons': self._generate_false_reasons(entity_check, numerical_check, keyword_check, semantic_sim, hallucination_score) if verdict == 'HALLUCINATION DETECTED' else [],
        }

    def _generate_computational_trace(self, entity_check, numerical_check, keyword_check, semantic_sim, hallucination_score) -> Dict:
        steps, contributions = [], []

        total_summary_entities = sum(len(v) for v in entity_check['summary_entities'].values())
        if total_summary_entities > 0:
            total_hall = sum(len(v) for v in entity_check['hallucinated'].values())
            rate = total_hall / total_summary_entities
            steps.append({
                'step': 1, 'name': 'Entity Hallucination Check',
                'detail': f'Found {total_summary_entities} entities: {total_hall} hallucinated, {total_summary_entities - total_hall} verified',
                'hallucinated': [e for ents in entity_check['hallucinated'].values() for e in ents],
                'verified':     [e for ents in entity_check['verified'].values() for e in ents],
                'formula': f'{total_hall} / {total_summary_entities} = {rate:.4f}',
                'contribution': rate, 'status': 'fail' if rate > 0 else 'pass',
            })
            contributions.append(rate)
        else:
            steps.append({'step': 1, 'name': 'Entity Hallucination Check',
                          'detail': 'No entities found — skipped', 'formula': 'N/A',
                          'contribution': None, 'status': 'skip'})

        total_nums = len(numerical_check['hallucinated']) + len(numerical_check['verified'])
        if total_nums > 0:
            num_rate = len(numerical_check['hallucinated']) / total_nums
            steps.append({
                'step': 2, 'name': 'Numerical Hallucination Check',
                'detail': f'Found {total_nums} numbers: {len(numerical_check["hallucinated"])} hallucinated, {len(numerical_check["verified"])} verified',
                'hallucinated': numerical_check['hallucinated'],
                'verified': numerical_check['verified'],
                'source_numbers': numerical_check['source_numbers'],
                'formula': f'{len(numerical_check["hallucinated"])} / {total_nums} = {num_rate:.4f}',
                'contribution': num_rate, 'status': 'fail' if num_rate > 0 else 'pass',
            })
            contributions.append(num_rate)
        else:
            steps.append({'step': 2, 'name': 'Numerical Hallucination Check',
                          'detail': 'No numbers found — skipped', 'formula': 'N/A',
                          'contribution': None, 'status': 'skip'})

        coverage = keyword_check['coverage_rate']
        kw_c = 1.0 - coverage
        steps.append({
            'step': 3, 'name': 'Keyword Coverage Check',
            'detail': f'{len(keyword_check["covered"])} of {len(keyword_check["covered"]) + len(keyword_check["missing"])} keywords covered ({coverage:.0%})',
            'covered': list(keyword_check['covered'])[:8],
            'missing': list(keyword_check['missing'])[:8],
            'formula': f'1 - {coverage:.4f} = {kw_c:.4f}',
            'contribution': kw_c, 'status': 'fail' if kw_c > 0.5 else 'pass',
        })
        contributions.append(kw_c)

        sim_c = 1.0 - semantic_sim
        steps.append({
            'step': 4, 'name': 'Semantic Similarity (TF-IDF Cosine)',
            'detail': f'Cosine similarity: {semantic_sim:.4f}',
            'formula': f'1 - {semantic_sim:.4f} = {sim_c:.4f}',
            'contribution': sim_c, 'status': 'fail' if sim_c > 0.5 else 'pass',
        })
        contributions.append(sim_c)

        total_checks = len(contributions)
        final = sum(contributions) / total_checks if total_checks else 0.0
        return {
            'steps': steps,
            'aggregation': {
                'formula': f'({" + ".join(f"{c:.4f}" for c in contributions)}) / {total_checks} = {final:.4f}',
                'contributions': contributions, 'total_checks': total_checks,
                'final_score': final, 'threshold': 0.35,
                'verdict': 'HALLUCINATION DETECTED' if final > 0.35 else 'SUMMARY VERIFIED',
            },
        }

    def _generate_false_reasons(self, entity_check, numerical_check, keyword_check, semantic_sim, hallucination_score) -> List[str]:
        reasons = []

        for ent_type, entities in entity_check['hallucinated'].items():
            for entity in entities:
                reasons.append(f'Entity "{entity}" (type: {ent_type}) appears in the summary but is absent from the source text')

        source_nums_str = ', '.join(numerical_check['source_numbers']) if numerical_check['source_numbers'] else 'none'
        for num in numerical_check['hallucinated']:
            reasons.append(f'Number "{num}" in the summary does not match any value in the source text (source contains: {source_nums_str})')

        if keyword_check['coverage_rate'] < 0.3:
            reasons.append(f'Only {keyword_check["coverage_rate"]:.0%} of source keywords represented — absent: {", ".join(list(keyword_check["missing"])[:5])}')

        if semantic_sim < 0.3:
            reasons.append(f'Low semantic similarity ({semantic_sim:.0%}) — summary diverges significantly from source')

        extra = list(keyword_check.get('extra', []))[:5]
        if extra:
            reasons.append(f'Summary introduces concepts not present in the source: {", ".join(extra)}')

        if not reasons:
            reasons.append(f'Overall hallucination score ({hallucination_score:.0%}) exceeds detection threshold (35%)')

        return reasons

    def _generate_explanation(self, entity_check, numerical_check, keyword_check, semantic_sim, hallucination_score) -> str:
        parts = ["=" * 70, "HALLUCINATION DETECTION REPORT", "=" * 70,
                 f"\nOverall Hallucination Score: {hallucination_score:.2%}",
                 f"Semantic Similarity to Source: {semantic_sim:.2%}",
                 f"Keyword Coverage: {keyword_check['coverage_rate']:.2%}", ""]

        if entity_check['hallucinated']:
            parts.append("⚠️  ENTITY HALLUCINATIONS DETECTED:")
            for ent_type, entities in entity_check['hallucinated'].items():
                parts += [f"   Type: {ent_type}", f"   Fabricated: {', '.join(entities)}",
                          f"   Reason: These entities do not appear in the source text", ""]

        if numerical_check['hallucinated']:
            parts += ["⚠️  NUMERICAL HALLUCINATIONS DETECTED:",
                      f"   Fabricated numbers: {', '.join(numerical_check['hallucinated'])}",
                      f"   Source numbers: {', '.join(numerical_check['source_numbers'])}",
                      f"   Reason: These numbers do not appear in the source text", ""]

        if not entity_check['hallucinated'] and not numerical_check['hallucinated']:
            parts += ["✓ No major hallucinations detected",
                      "  All entities and numbers appear to be grounded in the source"]

        parts.append("=" * 70)
        return "\n".join(parts)