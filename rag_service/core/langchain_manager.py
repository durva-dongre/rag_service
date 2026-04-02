# rag_service/rag_service/core/langchain_manager.py

import frappe
import json
from typing import Dict, List, Optional, Union
from datetime import datetime
from .llm_providers import create_llm_provider, OpenAIProvider

class LangChainManager:
    def __init__(self):
        self.llm = None
        self.llm_provider = None
        self.model_used = None
        self.setup_llm()

    def setup_llm(self):
        try:
            llm_settings = frappe.get_list(
                "LLM Settings",
                filters={"is_active": 1},
                limit=1
            )

            if not llm_settings:
                print("Warning: No active LLM configuration found — will retry on first use")
                return

            settings = frappe.get_doc("LLM Settings", llm_settings[0].name)
            self.model_used = llm_settings[0].name
            print(f"\nUsing LLM Settings:")
            print(f"Provider: {settings.provider}")
            print(f"Model: {settings.model_name}")

            self.llm_provider = create_llm_provider(
                provider=settings.provider,
                api_key=settings.get_password('api_secret'),
                model_name=settings.model_name,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens
            )

            if isinstance(self.llm_provider, OpenAIProvider):
                self.llm = self.llm_provider.llm

        except Exception as e:
            error_msg = f"LLM Setup Error: {str(e)}"
            print(f"\nError: {error_msg}")
            frappe.log_error(error_msg, "LLM Setup Error")

    def _ensure_llm(self):
        if not self.llm_provider:
            self.setup_llm()
        if not self.llm_provider:
            raise Exception("No active LLM configuration found")

    def clean_json_response(self, response: str) -> str:
        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                code_blocks = response.split("```")
                if len(code_blocks) >= 3:
                    response = code_blocks[1].strip()
                    if not (response.startswith('{') or response.startswith('[')):
                        json_start = response.find('{')
                        if json_start >= 0:
                            response = response[json_start:]

            if not response.strip().startswith('{'):
                json_start = response.find('{')
                if json_start >= 0:
                    response = response[json_start:]

            if not response.strip().endswith('}'):
                json_end = response.rfind('}')
                if json_end >= 0:
                    response = response[:json_end+1]

            return response.strip()
        except Exception as e:
            print(f"Error cleaning JSON: {str(e)}")
            return response

    def get_universal_template(self) -> Dict:
        try:
            print("\n=== Getting Universal Template ===")

            templates = frappe.get_list(
                "Prompt Template",
                filters={"is_active": 1},
                order_by="version desc",
                limit=1
            )

            if templates:
                template = frappe.get_doc("Prompt Template", templates[0].name)
                print(f"Using universal template: {template.template_name}")
                template.db_set('last_used', datetime.now())
                frappe.db.commit()
                return template
            else:
                print("No active template found, using built-in default")
                return self.get_builtin_template()

        except Exception as e:
            error_msg = f"Template Error: {str(e)}"
            print(f"\nError: {error_msg}")
            frappe.log_error(error_msg, "Template Error")
            return self.get_builtin_template()

    def get_builtin_template(self):
        class BuiltinTemplate:
            def __init__(self):
                self.template_name = "Built-in Universal Template"
                self.system_prompt = """You are an expert educational feedback assistant that provides constructive, age-appropriate feedback on student submissions across all subjects and assignment types. You adapt your evaluation criteria and language based on the assignment context provided.

                                        CRITICAL: You must ALWAYS respond with valid JSON, never plain text."""
                self.user_prompt = """Assignment Context:
                                        Assignment Name: {assignment_name}
                                        Subject Area: {course_vertical}
                                        Assignment Type: {assignment_type}
                                        Description: {assignment_description}

                                        Learning Objectives:
                                        {learning_objectives}

                                        Please analyze this student submission and provide feedback in the required JSON format."""
                self.response_format = """{
                                            "overall_feedback": "Comprehensive feedback about the submission",
                                            "strengths": ["Specific strength 1", "Specific strength 2", "Specific strength 3"],
                                            "areas_for_improvement": ["Improvement area 1", "Improvement area 2"],
                                            "learning_objectives_feedback": ["Feedback on learning objective 1"],
                                            "grade_recommendation": 85,
                                            "encouragement": "Encouraging message for the student"
                                        }"""

        return BuiltinTemplate()

    def format_objectives(self, objectives: List[Dict]) -> str:
        if not objectives:
            return "No specific learning objectives provided for this assignment."

        formatted = []
        for i, obj in enumerate(objectives, 1):
            if isinstance(obj, dict):
                description = obj.get('description', obj.get('objective_id', 'Unknown objective'))
            else:
                description = str(obj)
            formatted.append(f"{i}. {description}")

        return "\n".join(formatted)

    def get_default_response_format(self) -> Dict:
        return {
            "overall_feedback": "Overall assessment of the submission",
            "strengths": ["Strength 1", "Strength 2", "Strength 3"],
            "areas_for_improvement": ["Area 1", "Area 2"],
            "learning_objectives_feedback": ["Feedback on objective 1"],
            "grade_recommendation": 75,
            "encouragement": "Encouraging message for the student"
        }

    async def generate_feedback_universal(self, assignment_context: Dict, submission_url: str, submission_id: str) -> Dict:
        try:
            print("\n=== Starting Universal Feedback Generation ===")

            self._ensure_llm()

            template = self.get_universal_template()
            print("Template loaded successfully")

            try:
                if hasattr(template, 'response_format') and template.response_format:
                    expected_format = json.loads(template.response_format)
                    print("Using template-defined response format")
                else:
                    expected_format = self.get_default_response_format()
                    print("Using default response format")
            except json.JSONDecodeError:
                expected_format = self.get_default_response_format()
                print("Failed to parse template response format, using default")

            learning_objectives = self.format_objectives(assignment_context.get("learning_objectives", []))

            user_prompt_vars = {
                "assignment_name": assignment_context["assignment"].get("name", ""),
                "assignment_description": assignment_context["assignment"].get("description", ""),
                "course_vertical": assignment_context.get("course_vertical", "General"),
                "assignment_type": assignment_context["assignment"].get("type", "Practical"),
                "learning_objectives": learning_objectives
            }

            formatted_user_prompt = template.user_prompt
            for key, value in user_prompt_vars.items():
                placeholder = "{" + key + "}"
                if placeholder in formatted_user_prompt:
                    formatted_user_prompt = formatted_user_prompt.replace(placeholder, str(value))

            system_prompt = template.system_prompt

            messages = self.llm_provider.format_messages(
                system_prompt=system_prompt,
                user_prompt=formatted_user_prompt,
                image_url=submission_url
            )

            print(f"\nAssignment: {assignment_context['assignment'].get('name', 'Unknown')}")
            print(f"Subject: {assignment_context.get('course_vertical', 'General')}")
            print(f"Type: {assignment_context['assignment'].get('type', 'Unknown')}")
            print("\nSending request to LLM...")

            raw_text = await self.llm_provider.generate_with_vision(messages)
            print(f"\nRaw LLM Response: {raw_text}")

            try:
                cleaned_text = self.clean_json_response(raw_text)
                print(f"\nCleaned Response Text: {cleaned_text}")
                feedback = json.loads(cleaned_text)
                print("\nSuccessfully parsed JSON response")
                feedback = self.validate_feedback_structure(feedback, expected_format)
            except json.JSONDecodeError as e:
                print(f"\nJSON Parse Error: {str(e)}")
                print("Using fallback feedback format")
                feedback = self.create_fallback_feedback(assignment_context, expected_format)

            feedback["plagiarism_output"] = {
                "is_plagiarized": False,
                "is_ai_generated": False,
                "match_type": "original",
                "plagiarism_source": "none",
                "similarity_score": 0.0,
                "ai_detection_source": "none",
                "ai_confidence": 0.0,
                "similar_sources": []
            }

            try:
                template_used = template.name if hasattr(template, 'name') else "Built-in Universal Template"
            except Exception:
                template_used = "Built-in Universal Template"

            print("\n=== Feedback Generation Completed Successfully ===")
            return feedback, template_used

        except Exception as e:
            error_msg = f"Error generating feedback for submission {submission_id}: {str(e)}"
            print(f"\nError: {error_msg}")
            frappe.log_error(message=error_msg, title="Feedback Generation Error")
            return self.create_error_feedback(assignment_context), "Built-in Universal Template for Error"

    async def generate_feedback(self, assignment_context: Dict, submission_url: str, submission_id: str,
                                plagiarism_data: Dict = None, feedback_request_id: str = None) -> Dict:
        result_status = "Pending"

        try:
            if plagiarism_data:
                is_plagiarized = plagiarism_data.get("is_plagiarized", False)
                is_ai_generated = plagiarism_data.get("is_ai_generated", False)
                match_type = plagiarism_data.get("match_type", "original")

                if is_ai_generated:
                    result_status = "Success - Flagged"
                    feedback = self._create_ai_generated_feedback(plagiarism_data, assignment_context)
                    template_used = "Feedback Template for AI Generated Submission"
                elif is_plagiarized and match_type in ["exact_duplicate", "near_duplicate"]:
                    result_status = "Success - Flagged"
                    feedback = self._create_plagiarism_feedback(plagiarism_data, assignment_context)
                    template_used = "Feedback Template for Plagiarized Submission"
                else:
                    result_status = "Success - Original"
                    feedback, template_used = await self.generate_feedback_universal(assignment_context, submission_url, submission_id)

            await self._update_result_status(feedback_request_id, result_status)
            return feedback, self.model_used, template_used

        except Exception as e:
            result_status = "Failed"
            await self._update_result_status(feedback_request_id, result_status, str(e))
            raise

    async def _update_result_status(self, feedback_request_id: str, status: str, error_message: str = None):
        if not feedback_request_id:
            return

        update_data = {"result_status": status}
        if error_message:
            update_data["error_message"] = error_message[:500]

        frappe.db.set_value(
            "Feedback Request",
            feedback_request_id,
            update_data,
            update_modified=True
        )
        frappe.db.commit()

    def _create_ai_generated_feedback(self, plagiarism_data: Dict, assignment_context: Dict) -> Dict:
        ai_source = plagiarism_data.get("ai_detection_source", "unknown")
        ai_confidence = plagiarism_data.get("ai_confidence", 0.0)

        return {
            "overall_feedback": f"Your submission appears to be generated by an AI tool (detected source: {ai_source}, confidence: {ai_confidence:.0%}). At MentorMe, we encourage original creative work that reflects your own learning and artistic development. AI-generated images, while interesting, don't demonstrate the skills and creativity we're looking to nurture. Please submit your own original artwork for this assignment.",
            "strengths": ["N/A - AI-generated content detected"],
            "areas_for_improvement": ["Submit original artwork created by you", "Review assignment guidelines for creative direction"],
            "learning_objectives_feedback": ["Unable to assess - submission flagged as AI-generated"],
            "grade_recommendation": 0,
            "encouragement": "We believe in your creative abilities!",
            "plagiarism_output": {
                "is_plagiarized": False,
                "is_ai_generated": True,
                "match_type": "ai_generated",
                "plagiarism_source": "none",
                "similarity_score": 0.0,
                "ai_detection_source": ai_source,
                "ai_confidence": ai_confidence,
            }
        }

    def _create_plagiarism_feedback(self, plagiarism_data: Dict, assignment_context: Dict) -> Dict:
        match_type = plagiarism_data.get("match_type")
        plagiarism_source = plagiarism_data.get("plagiarism_source")
        similarity_score = plagiarism_data.get("similarity_score", 0.0)
        ai_confidence = plagiarism_data.get("ai_confidence", 0.0)

        return {
            "overall_feedback": f"Your submission has been flagged for similarity (similarity: {similarity_score:.0%}, source: {plagiarism_source}). Academic integrity is fundamental to the learning process. Please ensure your submissions represent your own original work.",
            "strengths": ["N/A - Submission flagged for similarity"],
            "areas_for_improvement": ["Create original artwork for this assignment", "Review academic integrity guidelines"],
            "learning_objectives_feedback": ["Unable to assess - submission flagged for similarity"],
            "grade_recommendation": 0,
            "encouragement": "Every artist develops their unique style through practice!",
            "plagiarism_output": {
                "is_plagiarized": True,
                "is_ai_generated": False,
                "match_type": match_type,
                "plagiarism_source": plagiarism_source,
                "similarity_score": similarity_score,
                "ai_detection_source": "none",
                "ai_confidence": ai_confidence,
            }
        }

    def validate_feedback_structure(self, feedback: Dict, expected_format: Dict) -> Dict:
        for field in expected_format:
            if field not in feedback:
                if isinstance(expected_format[field], list):
                    feedback[field] = ["No information provided"]
                elif isinstance(expected_format[field], (int, float)):
                    feedback[field] = 0
                else:
                    feedback[field] = "No information provided"

        try:
            grade = feedback.get("grade_recommendation", 0)
            if isinstance(grade, str):
                grade_clean = ''.join(c for c in grade if c.isdigit() or c == '.')
                grade = float(grade_clean) if grade_clean else 0
            feedback["grade_recommendation"] = max(0, min(100, float(grade)))
        except (ValueError, TypeError):
            feedback["grade_recommendation"] = 0

        for field in ["strengths", "areas_for_improvement", "learning_objectives_feedback"]:
            if field in feedback and not isinstance(feedback[field], list):
                feedback[field] = [str(feedback[field])]

        return feedback

    def create_fallback_feedback(self, assignment_context: Dict, expected_format: Dict) -> Dict:
        assignment_name = assignment_context["assignment"].get("name", "this assignment")

        fallback = {}
        for field, default_value in expected_format.items():
            if field == "overall_feedback":
                fallback[field] = f"I encountered a formatting issue while processing your submission for {assignment_name}. This appears to be a technical problem on our end. Please try resubmitting if this issue persists."
            elif field == "grade_recommendation":
                fallback[field] = 50
            elif isinstance(default_value, list):
                if "strength" in field:
                    fallback[field] = ["Your submission was received and processed"]
                elif "improvement" in field:
                    fallback[field] = ["Please ensure your submission clearly shows your work"]
                else:
                    fallback[field] = ["Unable to provide specific feedback due to processing issue"]
            else:
                if field == "encouragement":
                    fallback[field] = "Technical issues don't reflect your effort - please try resubmitting!"
                else:
                    fallback[field] = "Processing issue - please resubmit for detailed feedback"

        return fallback

    def create_error_feedback(self, assignment_context: Dict) -> Dict:
        assignment_name = assignment_context["assignment"].get("name", "this assignment")

        return {
            "overall_feedback": f"I encountered a system error while processing your submission for {assignment_name}. This appears to be a technical issue on our end. Please try resubmitting, and if the issue persists, contact your instructor.",
            "strengths": ["Your submission was received successfully"],
            "areas_for_improvement": ["No issues identified with your submission - this appears to be a technical problem"],
            "learning_objectives_feedback": ["Unable to evaluate due to system error - please resubmit"],
            "grade_recommendation": 0,
            "encouragement": "Technical issues don't reflect your effort or ability - please try again!"
        }

    @staticmethod
    def format_feedback_for_display(feedback: Dict) -> str:
        try:
            formatted = []

            if "overall_feedback" in feedback:
                formatted.append("Overall Feedback:")
                formatted.append(feedback["overall_feedback"])

            if "strengths" in feedback:
                formatted.append("\nStrengths:")
                for strength in feedback["strengths"]:
                    formatted.append(f"- {strength}")

            if "areas_for_improvement" in feedback:
                formatted.append("\nAreas for Improvement:")
                for area in feedback["areas_for_improvement"]:
                    formatted.append(f"- {area}")

            if "learning_objectives_feedback" in feedback:
                formatted.append("\nLearning Objectives Feedback:")
                for obj in feedback["learning_objectives_feedback"]:
                    formatted.append(f"- {obj}")

            if "grade_recommendation" in feedback:
                formatted.append(f"\nGrade Recommendation: {feedback['grade_recommendation']}")

            if "encouragement" in feedback:
                formatted.append(f"\nEncouragement: {feedback['encouragement']}")

            return "\n".join(formatted)

        except Exception as e:
            print(f"\nError: Error formatting feedback: {str(e)}")
            return "Error formatting feedback for display. Please check the JSON feedback data."

    def get_current_config(self) -> Dict:
        if not self.llm_provider:
            return {"status": "not_configured"}

        return {
            "provider": self.llm_provider.__class__.__name__,
            "model": self.llm_provider.model_name,
            "temperature": self.llm_provider.temperature,
            "max_tokens": self.llm_provider.max_tokens
        }
