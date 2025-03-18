from typing import Dict, List, Optional, Any, Union
import os
import json
import re
from enum import Enum

# Enums for content parameters
class ContentType(Enum):
    BLOG = "blog"
    ESSAY = "essay"
    TECHNICAL = "technical"
    NEWSLETTER = "newsletter"
    
class StyleType(Enum):
    FORMAL = "formal"
    CASUAL = "casual"
    ACADEMIC = "academic"
    CONVERSATIONAL = "conversational"
    PROFESSIONAL = "professional"
    
class ToneType(Enum):
    INFORMATIVE = "informative"
    PERSUASIVE = "persuasive"
    ENTERTAINING = "entertaining"
    AUTHORITATIVE = "authoritative"
    FRIENDLY = "friendly"
    ANALYTICAL = "analytical"
    
class FormatType(Enum):
    STANDARD = "standard"
    LISTICLE = "listicle"
    GUIDE = "guide"
    REVIEW = "review"
    OPINION = "opinion"
    COMPARISON = "comparison"

# Base prompt templates for different content types
BASE_TEMPLATES = {
    ContentType.BLOG.value: {
        "template": """Write a blog post about {topic}. The post should be {tone} in nature 
        and written in a {style} style. Format the content as a {format} piece.
        
        Structure the blog with clear sections including an introduction, main points, and conclusion.
        Each main point should be thoroughly explained with examples or evidence where appropriate.
        Use appropriate subheadings to organize the content and maintain reader engagement.
        
        The blog should have a conversational flow and address the reader directly when appropriate.
        Ensure the content is original, engaging, and provides value to the reader.
        
        {additional_instructions}
        """,
        "default_style": StyleType.CONVERSATIONAL.value,
        "default_tone": ToneType.INFORMATIVE.value,
        "default_format": FormatType.STANDARD.value
    },
    
    ContentType.ESSAY.value: {
        "template": """Write an essay on {topic}. The essay should adopt a {tone} tone 
        and be written in a {style} style. Structure the essay as a {format} piece.
        
        Begin with a clear thesis statement that outlines the main argument or perspective.
        Develop the argument through logical progression of ideas, with each paragraph focusing on a specific point.
        Use evidence, examples, or authoritative sources to support the main arguments.
        
        Maintain a coherent structure with clear transitions between paragraphs and sections.
        Conclude by summarizing the key points and reinforcing the thesis.
        
        {additional_instructions}
        """,
        "default_style": StyleType.ACADEMIC.value,
        "default_tone": ToneType.ANALYTICAL.value,
        "default_format": FormatType.STANDARD.value
    },
    
    ContentType.TECHNICAL.value: {
        "template": """Write a technical article about {topic}. The article should be {tone} 
        and written in a {style} style. Format the content as a {format} piece.
        
        Begin with an overview that explains the relevance and importance of the topic.
        Break down complex concepts into clear, accessible explanations.
        Include specific examples, code snippets, or diagrams where appropriate.
        
        Use technical terminology appropriately, defining terms when necessary.
        Structure the content with logical progression from basic to advanced concepts.
        Conclude with practical applications or future implications of the topic.
        
        {additional_instructions}
        """,
        "default_style": StyleType.PROFESSIONAL.value,
        "default_tone": ToneType.INFORMATIVE.value,
        "default_format": FormatType.GUIDE.value
    },
    
    ContentType.NEWSLETTER.value: {
        "template": """Create a newsletter issue about {topic}. The newsletter should have a {tone} tone 
        and be written in a {style} style. Format the content as a {format} piece.
        
        Begin with a brief greeting and introduction to the main topics covered in this issue.
        Organize the content into distinct sections with clear headings.
        Include relevant updates, insights, or news related to the main topic.
        
        Maintain a consistent voice throughout the newsletter that aligns with the brand identity.
        End with a call-to-action or preview of upcoming content.
        
        {additional_instructions}
        """,
        "default_style": StyleType.PROFESSIONAL.value,
        "default_tone": ToneType.FRIENDLY.value,
        "default_format": FormatType.STANDARD.value
    }
}

# Style modifier templates to adjust the base templates
STYLE_MODIFIERS = {
    StyleType.FORMAL.value: "Use formal language with proper grammar and vocabulary. Avoid contractions, "
                         "slang, or colloquialisms. Maintain a professional and objective tone throughout.",
    
    StyleType.CASUAL.value: "Use a relaxed, conversational language style. Feel free to use contractions, "
                         "personal anecdotes, and occasional informal expressions. Write as if speaking "
                         "directly to a friend while still being clear and coherent.",
    
    StyleType.ACADEMIC.value: "Employ scholarly language with precise terminology. Construct well-developed "
                           "arguments supported by evidence and references. Use complex sentence structures "
                           "and sophisticated vocabulary where appropriate.",
    
    StyleType.CONVERSATIONAL.value: "Write in a dialogue-like style that engages the reader. Ask rhetorical "
                                  "questions, use second-person pronouns, and adopt a friendly, accessible tone. "
                                  "Keep sentences relatively short and flow natural.",
    
    StyleType.PROFESSIONAL.value: "Maintain a business-appropriate tone that is clear, concise, and solution-focused. "
                               "Use industry-standard terminology without unnecessary jargon. Balance "
                               "being authoritative with being accessible."
}

# Tone modifier templates
TONE_MODIFIERS = {
    ToneType.INFORMATIVE.value: "Focus on providing clear, factual information. Present data, examples, "
                             "and explanations objectively. Prioritize clarity and comprehensiveness "
                             "over persuasion or entertainment.",
    
    ToneType.PERSUASIVE.value: "Craft compelling arguments designed to convince the reader. Use rhetorical "
                            "techniques, emotional appeals, and logical reasoning. Address potential "
                            "counterarguments and emphasize the benefits of adopting your perspective.",
    
    ToneType.ENTERTAINING.value: "Keep the content engaging and enjoyable to read. Use humor, storytelling, "
                              "or interesting examples where appropriate. Maintain a lively pace and "
                              "incorporate creative elements that capture attention.",
    
    ToneType.AUTHORITATIVE.value: "Establish expertise and credibility throughout the content. Make definitive "
                               "statements backed by evidence or experience. Project confidence in your "
                               "assertions and analysis.",
    
    ToneType.FRIENDLY.value: "Create a warm, approachable tone that builds rapport with readers. Use "
                         "inclusive language, show empathy, and acknowledge shared experiences. "
                         "Be encouraging and positive in your messaging.",
    
    ToneType.ANALYTICAL.value: "Take a methodical approach to examining the topic. Break down concepts "
                            "into constituent parts, evaluate evidence critically, and present logical "
                            "relationships between ideas. Maintain intellectual rigor throughout."
}

# Format modifier templates
FORMAT_MODIFIERS = {
    FormatType.STANDARD.value: "Structure with a clear introduction, body paragraphs, and conclusion. "
                           "Use transitions between sections and maintain a logical flow throughout.",
    
    FormatType.LISTICLE.value: "Organize as a numbered or bulleted list of key points. Each point should "
                           "have a clear heading followed by supporting paragraphs. Make each section "
                           "stand-alone while contributing to the overall topic.",
    
    FormatType.GUIDE.value: "Format as a step-by-step instruction or comprehensive reference. Include "
                        "practical advice, examples, and clear directions. Consider adding tips, "
                        "warnings, or additional resources where helpful.",
    
    FormatType.REVIEW.value: "Structure around evaluation of specific aspects of the subject. Include "
                         "both objective assessment and subjective opinion with supporting evidence. "
                         "Consider using a rating system or summary of pros and cons.",
    
    FormatType.OPINION.value: "Present a clear perspective or argument on the topic. Back opinions with "
                          "reasoning and evidence while acknowledging it is a viewpoint. Use persuasive "
                          "techniques to convince readers of your position.",
    
    FormatType.COMPARISON.value: "Organize content to contrast different aspects, options, or perspectives. "
                            "Use parallel structure to compare similar elements. Consider using tables, "
                            "charts, or side-by-side analysis for clarity."
}

class TemplateManager:
    """Manages content generation templates for different types of writing."""
    
    def __init__(self, templates_dir: str = None):
        """
        Initialize the template manager.
        
        Args:
            templates_dir: Optional directory containing custom templates
        """
        self.base_templates = BASE_TEMPLATES.copy()
        self.style_modifiers = STYLE_MODIFIERS.copy()
        self.tone_modifiers = TONE_MODIFIERS.copy()
        self.format_modifiers = FORMAT_MODIFIERS.copy()
        
        # Load custom templates if provided
        if templates_dir and os.path.exists(templates_dir):
            self._load_custom_templates(templates_dir)
    
    def _load_custom_templates(self, templates_dir: str) -> None:
        """
        Load custom templates from a directory.
        
        Args:
            templates_dir: Directory containing template JSON files
        """
        for filename in os.listdir(templates_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(templates_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        custom_templates = json.load(f)
                        
                        # Update templates if found in the file
                        if "base_templates" in custom_templates:
                            self.base_templates.update(custom_templates["base_templates"])
                        if "style_modifiers" in custom_templates:
                            self.style_modifiers.update(custom_templates["style_modifiers"])
                        if "tone_modifiers" in custom_templates:
                            self.tone_modifiers.update(custom_templates["tone_modifiers"])
                        if "format_modifiers" in custom_templates:
                            self.format_modifiers.update(custom_templates["format_modifiers"])
                except Exception as e:
                    print(f"Error loading template file {filename}: {e}")
    
    def get_template(
        self, 
        content_type: str, 
        style: str = None, 
        tone: str = None, 
        format_type: str = None,
        additional_instructions: str = ""
    ) -> str:
        """
        Get a formatted template based on content type and modifiers.
        
        Args:
            content_type: Type of content (blog, essay, technical, etc.)
            style: Writing style to use
            tone: Tone of the content
            format_type: Format structure for the content
            additional_instructions: Any additional specific instructions
            
        Returns:
            Formatted template string
        """
        # Validate and get content type template
        if content_type not in self.base_templates:
            # Default to blog if not found
            content_type = ContentType.BLOG.value
            
        base_template = self.base_templates[content_type]
        
        # Use default values if not specified
        style = style or base_template["default_style"]
        tone = tone or base_template["default_tone"]
        format_type = format_type or base_template["default_format"]
        
        # Get style, tone, and format modifiers
        style_modifier = self.style_modifiers.get(style, "")
        tone_modifier = self.tone_modifiers.get(tone, "")
        format_modifier = self.format_modifiers.get(format_type, "")
        
        # Combine the modifiers into additional instructions
        modifiers = f"Style guidelines: {style_modifier}\n\nTone guidelines: {tone_modifier}\n\nFormat guidelines: {format_modifier}"
        
        # Add user-provided additional instructions if any
        if additional_instructions:
            modifiers += f"\n\nAdditional specific instructions: {additional_instructions}"
        
        # Format the template with the modifiers
        template = base_template["template"].format(
            topic="{topic}",  # Leave as a placeholder for later formatting
            style=style,
            tone=tone,
            format=format_type,
            additional_instructions=modifiers
        )
        
        return template
    
    def format_with_examples(
        self,
        template: str,
        topic: str,
        examples: List[str] = None,
        num_examples: int = 2
    ) -> str:
        """
        Format a template with topic and optional few-shot examples.
        
        Args:
            template: Template string with placeholders
            topic: Main topic of the content
            examples: List of example texts to include as references
            num_examples: Number of examples to include (if available)
            
        Returns:
            Complete prompt with examples
        """
        # Format the template with the topic
        formatted = template.format(topic=topic)
        
        # Add examples if provided
        if examples and len(examples) > 0:
            # Limit to requested number of examples
            selected_examples = examples[:min(num_examples, len(examples))]
            
            examples_text = "\n\nHere are some examples of similar content for reference:\n\n"
            for i, example in enumerate(selected_examples, 1):
                # Truncate long examples to a reasonable preview
                preview = example[:1000] + "..." if len(example) > 1000 else example
                examples_text += f"Example {i}:\n{preview}\n\n"
                
            formatted += examples_text
            
            # Add final instruction to maintain consistent style with examples
            formatted += "\nMaintain a similar writing style, tone, and structure as shown in the examples while addressing the specific topic."
        
        return formatted
    
    def register_custom_template(self, template_type: str, template_name: str, template_data: Dict[str, Any]) -> None:
        """
        Register a new custom template.
        
        Args:
            template_type: Type of template ("base_templates", "style_modifiers", etc.)
            template_name: Name/key for the template
            template_data: Template data
        """
        if template_type == "base_templates":
            self.base_templates[template_name] = template_data
        elif template_type == "style_modifiers":
            self.style_modifiers[template_name] = template_data
        elif template_type == "tone_modifiers":
            self.tone_modifiers[template_name] = template_data
        elif template_type == "format_modifiers":
            self.format_modifiers[template_name] = template_data
        else:
            raise ValueError(f"Unknown template type: {template_type}")