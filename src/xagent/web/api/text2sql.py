"""Text2SQL database management API routes"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy.sql import func

from ..auth_dependencies import get_current_user
from ..models.database import get_db
from ..models.text2sql import DatabaseStatus, DatabaseType, Text2SQLDatabase
from ..models.user import User

# mypy: ignore-errors

logger = logging.getLogger(__name__)

# Create router
text2sql_router = APIRouter(prefix="/api/text2sql", tags=["text2sql"])


# Pydantic schemas
class DatabaseCreateRequest(BaseModel):
    """Request schema for creating a new database configuration"""

    name: str = Field(
        ..., min_length=1, max_length=255, description="Database display name"
    )
    type: str = Field(
        ..., description="Database type (sqlite, postgresql, mysql, sqlserver)"
    )
    url: str = Field(..., min_length=1, description="Database connection URL")
    read_only: bool = Field(default=True, description="Whether database is read-only")


class DatabaseResponse(BaseModel):
    """Response schema for database configuration"""

    id: int
    name: str
    type: str
    url: str
    read_only: bool
    status: str
    table_count: Optional[int] = None
    last_connected_at: Optional[str] = None
    error_message: Optional[str] = None
    created_at: str
    updated_at: str


class DataMapping(BaseModel):
    """Data mapping for chart axes"""

    xAxis: Optional[str] = None
    yAxis: Optional[str] = None
    valueAxis: Optional[str] = None


class ChartData(BaseModel):
    """Chart data structure"""

    columns: List[str]
    rows: List[Dict[str, Any]]


class PredictionRequest(BaseModel):
    """Request schema for data prediction"""

    chartType: str = Field(..., description="Chart type: bar, pie, line")
    data: ChartData = Field(..., description="Chart data")
    mapping: Optional[DataMapping] = Field(None, description="Data mapping for axes")
    predictPeriods: int = Field(default=5, description="Number of periods to predict")


class PredictionPoint(BaseModel):
    """Single prediction data point"""

    period: str
    predictedValue: float
    confidenceLower: Optional[float] = None
    confidenceUpper: Optional[float] = None


class PredictionResponse(BaseModel):
    """Response schema for data prediction"""

    success: bool
    predictedData: List[PredictionPoint]
    chartType: str
    confidence: Optional[str] = None
    trendAnalysis: Optional[str] = None
    error: Optional[str] = None


@text2sql_router.get("/databases", response_model=List[DatabaseResponse])
async def get_databases(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> List[DatabaseResponse]:
    """Get user's database configurations"""
    try:
        databases = (
            db.query(Text2SQLDatabase)
            .filter(Text2SQLDatabase.user_id == user.id)
            .order_by(Text2SQLDatabase.created_at.desc())
            .all()
        )

        return [DatabaseResponse(**db.to_dict()) for db in databases]
    except Exception as e:
        logger.error(f"Failed to get databases for user {user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve database configurations",
        )


@text2sql_router.post("/databases", response_model=DatabaseResponse)
async def create_database(
    db_config: DatabaseCreateRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> DatabaseResponse:
    """Create a new database configuration"""
    try:
        # Validate database type
        try:
            db_type = DatabaseType(db_config.type)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid database type: {db_config.type}",
            )

        # Check if user already has a database with the same name
        existing_db = (
            db.query(Text2SQLDatabase)
            .filter(
                Text2SQLDatabase.user_id == user.id,
                Text2SQLDatabase.name == db_config.name,
            )
            .first()
        )

        if existing_db:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Database with name '{db_config.name}' already exists",
            )

        # Create new database configuration
        new_db = Text2SQLDatabase(
            user_id=user.id,
            name=db_config.name,
            type=db_type,
            url=db_config.url,
            read_only=db_config.read_only,
            status=DatabaseStatus.CONNECTED,  # Set to connected by default
            table_count=0,  # TODO: Query actual table count
            last_connected_at=func.now(),
        )

        db.add(new_db)
        db.commit()
        db.refresh(new_db)

        logger.info(
            f"Created new database configuration for user {user.id}: {new_db.name}"
        )

        return DatabaseResponse(**new_db.to_dict())
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create database configuration: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create database configuration",
        )


@text2sql_router.put("/databases/{database_id}", response_model=DatabaseResponse)
async def update_database(
    database_id: int,
    db_config: DatabaseCreateRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> DatabaseResponse:
    """Update an existing database configuration"""
    try:
        # Get existing database
        existing_db = (
            db.query(Text2SQLDatabase)
            .filter(
                Text2SQLDatabase.id == database_id,
                Text2SQLDatabase.user_id == user.id,
            )
            .first()
        )

        if not existing_db:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Database configuration not found",
            )

        # Validate database type
        try:
            db_type = DatabaseType(db_config.type)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid database type: {db_config.type}",
            )

        # Check for name conflicts (excluding current database)
        name_conflict = (
            db.query(Text2SQLDatabase)
            .filter(
                Text2SQLDatabase.user_id == user.id,
                Text2SQLDatabase.name == db_config.name,
                Text2SQLDatabase.id != database_id,
            )
            .first()
        )

        if name_conflict:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Database with name '{db_config.name}' already exists",
            )

        # Update database configuration
        existing_db.name = db_config.name
        existing_db.type = db_type
        existing_db.url = db_config.url
        existing_db.read_only = db_config.read_only
        existing_db.status = (
            DatabaseStatus.DISCONNECTED
        )  # Reset status to verify new configuration
        existing_db.error_message = None

        db.commit()
        db.refresh(existing_db)

        logger.info(f"Updated database configuration {database_id} for user {user.id}")

        return DatabaseResponse(**existing_db.to_dict())
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update database configuration {database_id}: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update database configuration",
        )


@text2sql_router.delete("/databases/{database_id}")
async def delete_database(
    database_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Dict[str, str]:
    """Delete a database configuration"""
    try:
        # Get existing database
        existing_db = (
            db.query(Text2SQLDatabase)
            .filter(
                Text2SQLDatabase.id == database_id,
                Text2SQLDatabase.user_id == user.id,
            )
            .first()
        )

        if not existing_db:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Database configuration not found",
            )

        db.delete(existing_db)
        db.commit()

        logger.info(f"Deleted database configuration {database_id} for user {user.id}")

        return {"message": "Database configuration deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete database configuration {database_id}: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete database configuration",
        )


@text2sql_router.post("/databases/{database_id}/test")
async def test_database_connection(
    database_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Test database connection"""
    try:
        # Get existing database
        existing_db = (
            db.query(Text2SQLDatabase)
            .filter(
                Text2SQLDatabase.id == database_id,
                Text2SQLDatabase.user_id == user.id,
            )
            .first()
        )

        if not existing_db:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Database configuration not found",
            )

        # Simple connection test
        try:
            import sqlite3

            db_url = existing_db.url

            if existing_db.type.value == "sqlite":
                # SQLite connection test
                if db_url.startswith("sqlite:///"):
                    db_path = db_url.replace("sqlite:///", "")
                    conn = sqlite3.connect(db_path)
                    conn.close()

                    # Get table count
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT count(*) FROM sqlite_master WHERE type='table'"
                    )
                    table_count = cursor.fetchone()[0]
                    conn.close()

                else:
                    # Other format SQLite URL
                    conn = sqlite3.connect(db_url.replace("sqlite://", ""))
                    conn.close()
                    table_count = 0
            else:
                # For other database types, skip actual connection test for now
                table_count = 0

            # Update connection status
            existing_db.status = DatabaseStatus.CONNECTED
            existing_db.table_count = table_count
            existing_db.error_message = None
            existing_db.last_connected_at = func.now()
            db.commit()

            return {
                "status": "connected",
                "message": f"Database connection successful. Found {table_count} tables.",
                "table_count": table_count,
            }

        except Exception as test_error:
            # Connection test failed
            existing_db.status = DatabaseStatus.ERROR
            existing_db.error_message = str(test_error)
            db.commit()

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database connection failed: {str(test_error)}",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to test database connection {database_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database connection test failed: {str(e)}",
        )


def create_llm_from_db(db: Session, user_id: int):
    """Create LLM instance from user's database configuration"""
    try:
        from ..services.llm_utils import resolve_llms_for_user

        default_llm, _, _, _ = resolve_llms_for_user(db=db, user_id=user_id)

        if default_llm:
            logger.info(f"Using database LLM: {default_llm.model_name}")
            return default_llm
        else:
            logger.error("No default LLM found in database for user")
            return None

    except Exception as e:
        logger.error(f"Failed to create LLM from database: {e}")
        return None


async def generate_llm_prediction(
    chart_data: ChartData,
    chart_type: str,
    predict_periods: int,
    mapping: Optional[DataMapping] = None,
    db: Optional[Session] = None,
    user_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Generate prediction using LLM"""
    if not db or not user_id:
        raise ValueError("Both db session and user_id are required for prediction")

    llm = create_llm_from_db(db, user_id)
    if not llm:
        raise ValueError(
            "No LLM available for prediction. Please configure a default LLM model."
        )

    # Prepare data analysis prompt
    data_summary = f"Chart Type: {chart_type}\n"
    data_summary += f"Columns: {chart_data.columns}\n"
    data_summary += f"Data Points: {len(chart_data.rows)}\n"

    if mapping:
        data_summary += f"Data Mapping: X-axis={mapping.xAxis}, Y-axis={mapping.yAxis}, Value-axis={mapping.valueAxis}\n"

    data_summary += "\nSample Data:\n"
    for i, row in enumerate(chart_data.rows[:10]):  # Show first 10 rows
        data_summary += f"  {i + 1}. {row}\n"

    if len(chart_data.rows) > 10:
        data_summary += f"  ... and {len(chart_data.rows) - 10} more rows\n"

    prediction_prompt = f"""
You are a data analysis expert. Please perform trend analysis and prediction based on the following data:

{data_summary}

Please analyze the data trend and predict values for the next {predict_periods} periods.

Analysis requirements:
1. Identify the main trend of the data (growth, decline, cyclical, stable, etc.)
2. Provide confidence level for the trend analysis
3. Predict values for the next {predict_periods} periods
4. Provide reasonable confidence intervals for each prediction (if possible)

Please return results in JSON format with the following fields:
- trendAnalysis: Trend analysis description
- confidence: Prediction confidence level (high/medium/low)
- predictedData: Array of predicted data, each element contains:
  - period: Time period description
  - predictedValue: Predicted value
  - confidenceLower: Confidence interval lower bound (optional)
  - confidenceUpper: Confidence interval upper bound (optional)

Example return format:
{{
  "trendAnalysis": "Data shows a steady growth trend with a monthly growth rate of approximately 10%",
  "confidence": "high",
  "predictedData": [
    {{
      "period": "next period",
      "predictedValue": 150.5,
      "confidenceLower": 140.2,
      "confidenceUpper": 160.8
    }}
  ]
}}

Notes:
1. If data is insufficient or trend is unclear, please lower confidence and explain the reason
2. Predicted values should be based on reasonable extrapolation from historical data
3. For non-time series data (e.g., pie charts, bar charts), please make reasonable predictions based on existing patterns
"""

    # Generate prediction using LLM
    response = await llm.chat([{"role": "user", "content": prediction_prompt}])

    # Check if response is None
    if response is None:
        raise ValueError("LLM returned None response")

    # Extract content from response (handle both dict and string responses)
    if isinstance(response, str):
        content = response
    else:
        content = response.get("content", str(response))

    # Try to extract JSON from response
    import json
    import re

    # Look for JSON pattern in the response
    json_pattern = r"\{[\s\S]*\}"
    matches = re.findall(json_pattern, content)

    if matches:
        # Try the last match (most likely to be complete)
        json_str = matches[-1]
        try:
            prediction_data = json.loads(json_str)

            # Ensure we have the right structure
            if "predictedData" not in prediction_data:
                prediction_data["predictedData"] = []

            return {"success": True, **prediction_data}

        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON from LLM response: {json_str}")
            raise ValueError("Unable to parse LLM prediction response")

    else:
        raise ValueError("LLM did not return valid JSON prediction format")


@text2sql_router.post("/predict", response_model=PredictionResponse)
async def predict_data(
    request: PredictionRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> PredictionResponse:
    """Generate prediction based on chart data"""
    try:
        logger.info(
            f"User {user.id} requesting prediction for {request.chartType} chart"
        )

        # Generate prediction using LLM
        prediction_result = await generate_llm_prediction(
            chart_data=request.data,
            chart_type=request.chartType,
            predict_periods=request.predictPeriods,
            mapping=request.mapping,
            db=db,
            user_id=user.id,
        )

        if prediction_result["success"]:
            # Convert to response format
            predicted_data = []
            for point in prediction_result["predictedData"]:
                predicted_data.append(
                    PredictionPoint(
                        period=point["period"],
                        predictedValue=point["predictedValue"],
                        confidenceLower=point.get("confidenceLower"),
                        confidenceUpper=point.get("confidenceUpper"),
                    )
                )

            return PredictionResponse(
                success=True,
                predictedData=predicted_data,
                chartType=request.chartType,
                confidence=prediction_result.get("confidence"),
                trendAnalysis=prediction_result.get("trendAnalysis"),
            )
        else:
            return PredictionResponse(
                success=False,
                predictedData=[],
                chartType=request.chartType,
                error=prediction_result.get("error", "Unknown prediction error"),
            )

    except Exception as e:
        logger.error(f"Prediction API error for user {user.id}: {e}")
        return PredictionResponse(
            success=False,
            predictedData=[],
            chartType=request.chartType,
            error=f"Prediction service error: {str(e)}",
        )
