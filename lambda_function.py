
```python
import logging
import os
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple
import boto3
from botocore.exceptions import ClientError
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class FinOpsReportManager:
    """
    Orchestrates the collection of AWS cost data, AI-driven analysis 
    via Amazon Bedrock, and delivery via Amazon SNS.
    """
    def __init__(self, region: str = 'ap-south-1'):
        self.region = region
        self.ce_client = boto3.client('ce', region_name=self.region)
        self.bedrock_client = boto3.client('bedrock-runtime', region_name=self.region)
        self.sns_client = boto3.client('sns', region_name=self.region)
        self.topic_arn = os.environ.get('SNS_TOPIC_ARN')

    def _calculate_window(self, days: int = 7) -> Tuple[str, str]:
        now = datetime.today()
        start = now - timedelta(days=days)
        return start.strftime('%Y-%m-%d'), now.strftime('%Y-%m-%d')

    def _fetch_raw_costs(self, start: str, end: str) -> List[Dict[str, Any]]:
        try:
            response = self.ce_client.get_cost_and_usage(
                TimePeriod={'Start': start, 'End': end},
                Granularity='DAILY',
                Metrics=['UnblendedCost'],
                GroupBy=[{'Type': 'DIMENSION', 'Key': 'SERVICE'}]
            )
            return response.get('ResultsByTime', [])
        except ClientError as e:
            logger.error(f"Cost Explorer API failure: {e}")
            raise

    def _structure_cost_data(self, raw_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        processed_data = []
        for entry in raw_results:
            day_start = entry['TimePeriod']['Start']
            service_costs = {}
            for group in entry.get('Groups', []):
                service_name = group['Keys'][0]
                amount = float(group['Metrics']['UnblendedCost']['Amount'])
                if amount > 0.001:
                    service_costs[service_name] = round(amount, 4)
            processed_data.append({"Date": day_start, "Services": service_costs})
        return processed_data

    def _generate_ai_analysis(self, costs: List[Dict[str, Any]], start: str, end: str) -> str:
        prompt_template = (
            f"You are an Expert AWS FinOps Architect. Analyze the last 7 days of AWS spend:\n"
            f"{json.dumps(costs)}\n\n"
            f"Write a daily AWS Cost Report email with these sections:\n\n"
            f"DAILY AWS FINOPS REPORT\n"
            f"Period: {start} to {end}\n"
            f"============================================================\n\n"
            f"1. SPEND SUMMARY\n(Total spend, daily trend, highest spend day)\n\n"
            f"2. TOP SERVICES BY COST\n(List each service, its total cost, and % of total bill)\n\n"
            f"3. ANOMALIES DETECTED\n(Any unusual spikes or patterns)\n\n"
            f"4. TOP 3 RECOMMENDATIONS\n(Specific, actionable steps to reduce cost)\n\n"
            f"IMPORTANT FORMATTING RULES:\n"
            f"- Use PLAIN TEXT ONLY. No markdown, no **, no ##, no asterisks.\n"
            f"- Use ALL CAPS for section headers.\n"
            f"- Use dashes (-) for bullet points.\n"
            f"- Separate sections with a line of dashes ---\n"
            f"- Keep it concise, professional, and easy to read in an email."
        )
        try:
            payload = {"messages": [{"role": "user", "content": prompt_template}]}
            response = self.bedrock_client.invoke_model(
                modelId='openai.gpt-oss-safeguard-120b',
                body=json.dumps(payload),
                contentType='application/json'
            )
            body = json.loads(response['body'].read().decode())
            content = body['choices'][0]['message']['content']
            cleaned_report = re.sub(r'<reasoning>.*?</reasoning>', '', content, flags=re.DOTALL)
            return cleaned_report.strip().lstrip(': \n')
        except (ClientError, KeyError, json.JSONDecodeError) as e:
            logger.error(f"Bedrock AI analysis failed: {e}")
            return "Error generating AI report: Analysis service unavailable."

    def _publish_to_sns(self, message: str) -> None:
        if not self.topic_arn:
            raise ValueError("SNS_TOPIC_ARN environment variable is missing.")
        try:
            self.sns_client.publish(
                TopicArn=self.topic_arn,
                Subject="Daily AI FinOps Report",
                Message=message
            )
        except ClientError as e:
            logger.error(f"SNS Publication failed: {e}")
            raise

    def execute(self) -> str:
        start_date, end_date = self._calculate_window()
        raw_data = self._fetch_raw_costs(start_date, end_date)
        refined_costs = self._structure_cost_data(raw_data)
        final_report = self._generate_ai_analysis(refined_costs, start_date, end_date)
        self._publish_to_sns(final_report)
        return "Report Sent"

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    try:
        manager = FinOpsReportManager()
        result = manager.execute()
        return {"statusCode": 200, "body": result}
    except Exception as e:
        logger.exception(f"Critical failure in FinOps Pipeline: {str(e)}")
        return {"statusCode": 500, "body": f"Internal Server Error: {str(e)}"}


```
