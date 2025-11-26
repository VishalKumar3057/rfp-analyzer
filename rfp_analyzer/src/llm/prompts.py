"""Prompt templates for RFP analysis."""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


class PromptTemplates:
    """Collection of prompt templates for RFP analysis."""

    SYSTEM_PROMPT = """You are an expert RFP (Request for Proposal) analyst. Your role is to:
1. Analyze RFP documents thoroughly and accurately
2. Extract requirements with precision
3. Identify relationships between different sections
4. Provide clear, actionable insights
5. Flag any ambiguities or potential issues

Always base your analysis on the provided document context. Be specific and cite sections when possible.
If information is unclear or missing, explicitly state the uncertainty."""

    REQUIREMENT_EXTRACTION_TEMPLATE = """Analyze the following RFP document sections and extract all requirements related to: {query}

Retrieved Document Sections:
{context}

Instructions:
1. Identify ALL requirements (explicit and implicit) related to the query
2. For each requirement, provide:
   - A unique identifier (e.g., REQ-001)
   - A clear title
   - Full description
   - The section where it was found
   - Priority level if mentioned (high/medium/low)
   - Category (technical/security/performance/compliance/etc.)
3. Identify any related requirements that should be considered together
4. Explain your reasoning for why these requirements are relevant

Provide your response in the following JSON format:
{{
    "extracted_requirements": [
        {{
            "requirement_id": "REQ-001",
            "title": "Brief title",
            "description": "Full description",
            "section": "Section name/number",
            "page_number": null,
            "priority": "high|medium|low|null",
            "category": "category name",
            "related_requirements": ["REQ-002"]
        }}
    ],
    "reasoning": "Detailed explanation of how you found these requirements and why they are relevant",
    "gaps_or_conflicts": ["Any gaps or conflicts identified"],
    "confidence": 85,
    "uncertainties": ["Any unclear aspects"]
}}"""

    GAP_ANALYSIS_TEMPLATE = """Analyze the RFP requirements against the proposed approach.

Proposed Approach:
{approach}

RFP Requirements from Retrieved Sections:
{context}

Instructions:
1. List all RFP requirements found in the context
2. For each requirement, assess if the proposed approach addresses it
3. Identify any gaps - requirements that the approach doesn't address
4. Identify any potential conflicts between the approach and requirements
5. Suggest how gaps might be addressed

Provide your response in JSON format:
{{
    "extracted_requirements": [...],
    "reasoning": "Analysis of how well the approach addresses requirements",
    "gaps_or_conflicts": ["Specific gaps and conflicts identified"],
    "confidence": 80,
    "uncertainties": ["Areas needing clarification"]
}}"""

    COMPLIANCE_CHECK_TEMPLATE = """Check if the proposed approach complies with RFP requirements.

Approach to Evaluate:
{approach}

Target Section: {section}

RFP Requirements:
{context}

Instructions:
1. Extract ALL requirements from the provided RFP sections as "extracted_requirements"
2. Evaluate each requirement against the proposed approach
3. Classify compliance as: COMPLIANT, PARTIAL, NON-COMPLIANT, or UNCLEAR
4. Provide specific reasoning for each classification
5. Suggest modifications needed for full compliance

IMPORTANT: You MUST extract all requirements found in the context as structured requirement objects.
Each requirement should be formatted with requirement_id, title, description, and section.

Response format:
{{
    "extracted_requirements": [
        {{
            "requirement_id": "REQ-001",
            "title": "Requirement title",
            "description": "Full requirement description with compliance status (COMPLIANT/PARTIAL/NON-COMPLIANT)",
            "section": "Section name",
            "category": "security|compliance|technical"
        }}
    ],
    "reasoning": "Detailed compliance analysis with status for each requirement",
    "gaps_or_conflicts": ["Non-compliant or partially compliant items"],
    "confidence": 75,
    "uncertainties": ["Requirements that need clarification"]
}}"""

    CONFLICT_DETECTION_TEMPLATE = """Analyze the RFP for internal conflicts and consistency issues.

Query Context:
Timeline: {timeline}
Budget: {budget}  
Scope: {scope}

RFP Document Sections:
{context}

Instructions:
1. Identify timeline requirements and check for consistency
2. Identify budget constraints and verify feasibility
3. Analyze scope against timeline and budget
4. Detect any contradictions within or across sections
5. Flag implicit conflicts that may not be immediately obvious

Response format:
{{
    "extracted_requirements": [...],
    "reasoning": "Analysis of consistency across timeline, budget, and scope",
    "gaps_or_conflicts": ["All identified conflicts and inconsistencies"],
    "confidence": 70,
    "uncertainties": ["Areas where information is insufficient to determine conflicts"]
}}"""

    AMBIGUITY_ANALYSIS_TEMPLATE = """Analyze the usage of ambiguous terms in the RFP.

Term to Analyze: {term}

RFP Document Sections:
{context}

Instructions:
1. Find all occurrences of the term or related concepts in the provided context
2. For each occurrence, extract it as a requirement with the specific meaning in that context
3. Identify any inconsistencies in how the term is used
4. Suggest clarifying questions that should be asked
5. Provide interpretation recommendations

IMPORTANT: You MUST extract ALL related requirements where the term or similar concepts appear.
Each requirement represents a different usage context of the ambiguous term.

Response format:
{{
    "extracted_requirements": [
        {{
            "requirement_id": "AMB-001",
            "title": "Context-specific interpretation",
            "description": "How the term is used in this specific context and its interpreted meaning",
            "section": "Section where this usage was found",
            "category": "interpretation"
        }}
    ],
    "reasoning": "Analysis of term usage across different contexts with interpretations",
    "gaps_or_conflicts": ["Inconsistencies in term usage"],
    "confidence": 65,
    "uncertainties": ["Contexts where meaning is unclear"]
}}"""

    GENERAL_ANALYSIS_TEMPLATE = """Analyze the RFP documents to answer the user's question.

User Question:
{query}

Retrieved Document Sections:
{context}

Instructions:
1. Carefully analyze the retrieved sections
2. Extract relevant requirements and information
3. Provide a clear, comprehensive answer to the question
4. Include specific references to sections when possible
5. Note any limitations or uncertainties in your analysis

Response format:
{{
    "extracted_requirements": [...],
    "reasoning": "Comprehensive answer with supporting evidence from the documents",
    "gaps_or_conflicts": [],
    "confidence": 80,
    "uncertainties": []
}}"""

    @classmethod
    def get_template(cls, template_name: str) -> str:
        """Get a specific template by name."""
        templates = {
            "requirement_extraction": cls.REQUIREMENT_EXTRACTION_TEMPLATE,
            "gap_analysis": cls.GAP_ANALYSIS_TEMPLATE,
            "compliance_check": cls.COMPLIANCE_CHECK_TEMPLATE,
            "conflict_detection": cls.CONFLICT_DETECTION_TEMPLATE,
            "ambiguity_analysis": cls.AMBIGUITY_ANALYSIS_TEMPLATE,
            "general": cls.GENERAL_ANALYSIS_TEMPLATE,
        }
        return templates.get(template_name, cls.GENERAL_ANALYSIS_TEMPLATE)

    @classmethod
    def get_chat_prompt(cls, template_name: str) -> ChatPromptTemplate:
        """Get a ChatPromptTemplate for a specific analysis type."""
        template = cls.get_template(template_name)
        return ChatPromptTemplate.from_messages([
            ("system", cls.SYSTEM_PROMPT),
            ("human", template),
        ])
