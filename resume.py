import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from datetime import datetime
import os
import re
from flask import Flask
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
import PyPDF2
import docx
import tempfile
import base64
import pandas as pd
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json

# Initialize Flask
server = Flask(__name__)
app = dash.Dash(__name__, server=server,
                external_stylesheets=[dbc.themes.BOOTSTRAP])

# Configure Flask app
server.config['SECRET_KEY'] = 'your_secret_key_here'
server.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///your_database.db'
server.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy with the Flask server instead of the Dash app
db = SQLAlchemy(server)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

server.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
server.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max file size

# Initialize database
Base = declarative_base()

# Database Model


class Candidate(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    date = db.Column(db.DateTime, default=datetime.now)
    email = db.Column(db.String(100))
    phone_number = db.Column(db.String(20))
    location = db.Column(db.String(200))
    cv_file = db.Column(db.String(255))
    current_status = db.Column(db.String(50), default='CV Review')
    status_due_date = db.Column(db.DateTime, nullable=True)
    assignee = db.Column(db.String(100), nullable=True)
    position = db.Column(db.String(100), nullable=True)
    notified = db.Column(db.Boolean, default=False)
    fail_stage = db.Column(db.String(50), nullable=True)
    failed_reason = db.Column(db.Text, nullable=True)
    source_notes = db.Column(db.Text, nullable=True)
    last_updated = db.Column(
        db.DateTime, default=datetime.now, onupdate=datetime.now)

    def __repr__(self):
        return f'<Candidate {self.name}>'

# File processing functions


def extract_text_from_file(file_path):
    """Extract text from PDF or DOCX using Langchain loaders and text splitter"""
    try:
        # Get appropriate loader
        if file_path.lower().endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.lower().endswith(('.doc', '.docx')):
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError("Unsupported file format")

        # Load documents
        documents = loader.load()

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        docs = text_splitter.split_documents(documents)

        # Combine all text
        return " ".join([doc.page_content for doc in docs])

    except Exception as e:
        print(f"Error extracting text: {str(e)}")
        raise


def extract_cv_info(text):
    # Initialize Groq LLM
    llm = ChatGroq(
        api_key="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",
        model_name="llama-3.3-70b-versatile"
    )

    # Create a more comprehensive prompt template
    template = """
    You are an expert CV analyzer. Extract the following information from the CV below in a structured format.
    If any field is not found, indicate with "Not specified".

    CV TEXT:
    {cv_text}

    EXTRACT THE FOLLOWING INFORMATION:
    1. Full Name
    2. Email Address
    3. Phone Number
    4. Location/Address
    5. Summary/Profile (keep brief)
    6. Skills (list main technical and soft skills)
    7. Work Experience (for each position: company, title, dates)
    8. Education (for each degree: institution, degree, dates)

    IMPORTANT: Your response must be a valid, parseable JSON object with the following format:
    {{
        "full_name": "String value",
        "email": "String value",
        "phone": "String value",
        "location": "String value",
        "summary": "String value",
        "skills": ["Skill 1", "Skill 2", ...],
        "work_experience": [
            {{
                "company": "String value",
                "title": "String value",
                "dates": "String value"
            }},
            ...
        ],
        "education": [
            {{
                "institution": "String value",
                "degree": "String value",
                "dates": "String value"
            }},
            ...
        ]
    }}
    DO NOT include ANY explanatory text before or after the JSON object.
    Your entire response must be ONLY valid, parseable JSON, nothing else.
    """

    prompt = PromptTemplate(
        input_variables=["cv_text"],
        template=template
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    try:
        # Get response from Groq
        result = chain.run(cv_text=text)

        # Try to extract JSON from the result if there's extra text
        import re
        json_match = re.search(r'({.*})', result, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            cv_data = json.loads(json_str)
        else:
            # If no JSON pattern found, try to parse the whole result
            cv_data = json.loads(result)

        # Return extracted information in the format expected by the rest of the application
        return {
            'name': cv_data.get('full_name', 'Not specified'),
            'email': cv_data.get('email', 'Not specified'),
            'phone': cv_data.get('phone', 'Not specified'),
            'location': cv_data.get('location', 'Not specified'),
            'summary': cv_data.get('summary', 'Not specified'),
            'skills': cv_data.get('skills', []),
            'work_experience': cv_data.get('work_experience', []),
            'education': cv_data.get('education', []),
            'first_part': text[:100]  # Keep this for compatibility
        }

    except Exception as e:
        print(f"Error extracting information: {str(e)}")
        print(f"Raw response: {result}")  # Add this for debugging
        return {
            'name': None,
            'email': None,
            'phone': None,
            'location': None,
            'summary': None,
            'skills': [],
            'work_experience': [],
            'education': [],
            'first_part': text[:100]
        }


# Initialize database tables
with server.app_context():
    db.drop_all()
    db.create_all()

# Navigation bar
navbar = dbc.Navbar(
    dbc.Container([
        dbc.NavbarBrand("Recruitment Portal", className="ms-2"),
        dbc.Nav([
            dbc.NavItem(dbc.NavLink("Home", href="#")),
            dbc.NavItem(dbc.NavLink("NSS Candidates", href="#")),
        ], className="me-auto")
    ]),
    dark=True,
    color="dark",
)

# File upload modal
upload_modal = dbc.Modal(
    [
        dbc.ModalHeader("Upload CV"),
        dbc.ModalBody([
            dcc.Upload(
                id='upload-cv',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select a PDF/Word File')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center'
                },
                accept='.pdf,.doc,.docx'
            ),
            html.Div(id='upload-output', className="mt-3")
        ]),
    ],
    id="upload-modal",
    is_open=False,
)

# Information form modal
info_modal = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle("Edit Information Extraction")),
        dbc.ModalBody([
            dbc.Row([
                dbc.Col([
                    dbc.Label("Name"),
                    dbc.Input(id="candidate-name",
                              placeholder="Information Extraction", type="text"),
                ], width=6),
                dbc.Col([
                    dbc.Label("Email"),
                    dbc.Input(id="candidate-email",
                              placeholder="example@email.com", type="email"),
                ], width=6),
            ], className="mb-3"),

            dbc.Row([
                dbc.Col([
                    dbc.Label("Phone Number"),
                    dbc.Input(id="candidate-phone",
                              placeholder="Phone Number", type="text"),
                ], width=6),
                dbc.Col([
                    dbc.Label("Position"),
                    dbc.Select(
                        id="candidate-position",
                        options=[
                            {"label": "Select Position", "value": ""},
                            {"label": "Software Engineer", "value": "se"},
                            {"label": "Data Scientist", "value": "ds"},
                            {"label": "Product Manager", "value": "pm"},
                        ],
                    ),
                ], width=6),
            ], className="mb-3"),

            dbc.Row([
                dbc.Col([
                    dbc.Label("Location"),
                    dbc.Input(id="candidate-location",
                              placeholder="Candidate Location", type="text"),
                ], width=12),
            ], className="mb-3"),

            dbc.Row([
                dbc.Col([
                    dbc.Label("Current Status"),
                    dbc.Select(
                        id="current-status",
                        options=[
                            {"label": "CV Review", "value": "review"},
                            {"label": "Interview", "value": "interview"},
                            {"label": "Offer", "value": "offer"},
                            {"label": "Rejected", "value": "rejected"},
                        ],
                    ),
                ], width=6),
                dbc.Col([
                    dbc.Label("Status Due Date"),
                    dbc.Input(id="status-due-date", type="date"),
                ], width=6),
            ], className="mb-3"),

            dbc.Row([
                dbc.Col([
                    dbc.Label("Assignee"),
                    dbc.Input(id="assignee",
                              placeholder="Assignee name", type="text"),
                ], width=12),
            ], className="mb-3"),

            dbc.Row([
                dbc.Col([
                    dbc.Label("Failed Stage"),
                    dbc.Select(
                        id="failed-stage",
                        options=[
                            {"label": "Not Failed", "value": "none"},
                            {"label": "CV Review", "value": "cv"},
                            {"label": "Interview", "value": "interview"},
                        ],
                    ),
                ], width=12),
            ], className="mb-3"),

            dbc.Row([
                dbc.Col([
                    dbc.Label("Failed Reason"),
                    dbc.Textarea(
                        id="failed-reason", placeholder="Up to 500 characters", maxLength=500),
                ], width=12),
            ], className="mb-3"),

            dbc.Row([
                dbc.Col([
                    dbc.Label("Source Notes"),
                    dbc.Textarea(
                        id="source-notes", placeholder="Up to 500 characters", maxLength=500),
                ], width=12),
            ], className="mb-3"),

            dbc.Row([
                dbc.Col([
                    dbc.Checkbox(
                        id="candidate-notified",
                        label="Candidate Notified",
                        value=False  # Initial value
                    ),
                ], width=12),
            ], className="mb-3"),

            dbc.Button("Save Changes", id="save-changes", color="primary"),
            dbc.Button("Back to List", id="back-to-list",
                       color="secondary", className="ms-2"),
        ]),
    ],
    id="info-modal",
    size="xl",
    is_open=False,
)

# Update main content to conditionally show the table
content = html.Div([
    html.Div([
        html.H1("Employee Candidates", className="d-inline-block"),
        dbc.Button("Upload New CV", id="open-upload",
                   color="primary", className="float-end"),
    ], className="d-flex justify-content-between align-items-center mb-4"),

    html.Div(id="alert-container"),  # Container for the alert
    html.Div(id="table-container"),  # Container for the table
    upload_modal,
    info_modal
])

# Layout
app.layout = html.Div([
    navbar,
    dbc.Container(content, className="mt-4")
])

# Combined callback for modals, table updates, and alerts


@app.callback(
    [Output("upload-modal", "is_open"),
     Output("info-modal", "is_open"),
     Output("table-container", "children"),
     Output("alert-container", "children", allow_duplicate=True)],
    [Input("open-upload", "n_clicks"),
     Input("upload-cv", "contents"),
     Input("save-changes", "n_clicks"),
     Input({"type": "edit-btn", "index": dash.ALL}, "n_clicks"),
     Input({"type": "delete-btn", "index": dash.ALL}, "n_clicks")],
    [State("upload-cv", "filename"),
     State("upload-modal", "is_open"),
     State("info-modal", "is_open"),
     State("candidate-name", "value"),
     State("candidate-email", "value"),
     State("candidate-phone", "value"),
     State("candidate-position", "value"),
     State("current-status", "value"),
     State("status-due-date", "value"),
     State("assignee", "value"),
     State("failed-stage", "value"),
     State("failed-reason", "value"),
     State("source-notes", "value"),
     State("candidate-notified", "value"),
     State("candidate-location", "value"),
     State("table-container", "children")],
    prevent_initial_call=True
)
def handle_all_updates(btn_clicks, contents, save_clicks, edit_clicks, delete_clicks,
                       filename, upload_is_open, info_is_open,
                       name, email, phone, position, status, due_date, assignee,
                       fail_stage, failed_reason, source_notes, notified, location,
                       current_table):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    triggered_id = ctx.triggered[0]["prop_id"]

    try:
        # Handle Upload New CV button
        if "open-upload" in triggered_id:
            return True, False, dash.no_update, None

        # Handle file upload
        if "upload-cv" in triggered_id and contents is not None:
            try:
                # Create a temporary file to process
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
                    content_type, content_string = contents.split(',')
                    decoded = base64.b64decode(content_string)
                    temp_file.write(decoded)
                    temp_file_path = temp_file.name

                # Extract text using Langchain
                text = extract_text_from_file(temp_file_path)

                # Get parsed information from Groq API
                parsed_info = extract_cv_info(text)

                if parsed_info and any([parsed_info['name'], parsed_info['email'], parsed_info['phone'], parsed_info['location']]):
                    # Save the file permanently
                    secure_name = secure_filename(filename)
                    permanent_path = os.path.join(
                        server.config['UPLOAD_FOLDER'], secure_name)
                    os.rename(temp_file_path, permanent_path)

                    # Create new candidate in database
                    new_candidate = Candidate(
                        name=parsed_info['name'],
                        email=parsed_info['email'],
                        phone_number=parsed_info['phone'],
                        location=parsed_info['location'],
                        cv_file=secure_name,
                    )

                    db.session.add(new_candidate)
                    db.session.commit()

                    # Create table row with the extracted information
                    current_date = datetime.now().strftime("%Y-%m-%d")
                    new_row = html.Tr([
                        html.Td(parsed_info['name'] or "Not specified"),
                        html.Td(parsed_info['email'] or "Not specified"),
                        html.Td(parsed_info['phone'] or "Not specified"),
                        html.Td(parsed_info['location'] or "Not specified"),
                        html.Td(current_date),
                        html.Td("CV Review"),
                        html.Td("Not set"),
                        html.Td("Unassigned"),
                        html.Td("Not specified"),
                        html.Td([
                            dbc.Button("Edit", id={"type": "edit-btn", "index": 0},
                                       color="primary", size="sm", className="me-2"),
                            dbc.Button("Delete", id={"type": "delete-btn", "index": 0},
                                       color="danger", size="sm"),
                        ])
                    ])

                    # Create or update table
                    if current_table is None:
                        table = dbc.Table(
                            [
                                html.Thead(
                                    html.Tr([
                                        html.Th(col) for col in [
                                            "Name", "Email", "Phone", "Location", "Date Uploaded",
                                            "Current Status", "Status Due Date",
                                            "Assignee", "Position", "Actions"
                                        ]
                                    ])
                                ),
                                html.Tbody([new_row])
                            ],
                            bordered=True,
                            hover=True,
                            className="mt-4"
                        )
                        return False, True, table, dbc.Alert(
                            "CV processed successfully! Please review the extracted information.",
                            color="success",
                            dismissable=True,
                            duration=3000
                        )
                    else:
                        existing_rows = current_table["props"]["children"][1]["props"]["children"]
                        next_index = len(existing_rows)
                        new_row.children[-1].children[0].id["index"] = next_index
                        new_row.children[-1].children[1].id["index"] = next_index

                        table = dbc.Table(
                            [
                                current_table["props"]["children"][0],
                                html.Tbody(existing_rows + [new_row])
                            ],
                            bordered=True,
                            hover=True,
                            className="mt-4"
                        )
                        return False, True, table, dbc.Alert(
                            "CV processed successfully! Please review the extracted information.",
                            color="success",
                            dismissable=True,
                            duration=3000
                        )
                else:
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                    return False, True, dash.no_update, dbc.Alert(
                        "Could not extract information from the CV. Please fill in manually.",
                        color="warning",
                        dismissable=True,
                        duration=3000
                    )

            except Exception as e:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                print(f"Error processing file: {str(e)}")  # For debugging
                return False, False, dash.no_update, dbc.Alert(
                    f"Error processing file: {str(e)}",
                    color="danger",
                    dismissable=True,
                    duration=3000
                )

        # Handle Edit/Delete buttons
        elif isinstance(eval(triggered_id.split('.')[0]), dict):
            trigger_dict = eval(triggered_id.split('.')[0])
            if trigger_dict["type"] == "edit-btn":
                return False, True, dash.no_update, None
            elif trigger_dict["type"] == "delete-btn":
                if current_table:
                    try:
                        index = trigger_dict["index"]
                        # Delete from database
                        candidates = Candidate.query.all()
                        if index < len(candidates):
                            candidate_to_delete = candidates[index]
                            db.session.delete(candidate_to_delete)
                            db.session.commit()

                        # Update table
                        existing_rows = current_table["props"]["children"][1]["props"]["children"]
                        updated_rows = [row for i, row in enumerate(
                            existing_rows) if i != index]

                        if not updated_rows:
                            return False, False, None, dbc.Alert(
                                "No candidates in the database.",
                                color="info",
                                dismissable=True,
                                duration=3000
                            )

                        # Update indices for remaining rows
                        for i, row in enumerate(updated_rows):
                            row.children[-1].children[0].id["index"] = i
                            row.children[-1].children[1].id["index"] = i

                        updated_table = dbc.Table(
                            [
                                current_table["props"]["children"][0],
                                html.Tbody(updated_rows)
                            ],
                            bordered=True,
                            hover=True,
                            className="mt-4"
                        )
                        return False, False, updated_table, dbc.Alert(
                            "Entry deleted successfully!",
                            color="success",
                            dismissable=True,
                            duration=3000
                        )
                    except Exception as e:
                        print(f"Error deleting record: {str(e)}")
                        return False, False, dash.no_update, dbc.Alert(
                            f"Error deleting record: {str(e)}",
                            color="danger",
                            dismissable=True,
                            duration=3000
                        )

        # Handle Save Changes
        elif "save-changes" in triggered_id:
            if not any([name, email, phone]):  # Basic validation
                return False, True, dash.no_update, dbc.Alert(
                    "Please fill in at least some basic information.",
                    color="warning",
                    dismissable=True,
                    duration=3000
                )

            try:
                # Get all candidates and rebuild the table
                candidates = Candidate.query.all()
                table_rows = []

                for i, candidate in enumerate(candidates):
                    # If this is the latest candidate, use the form values
                    if i == len(candidates) - 1:
                        # Update the database record
                        candidate.name = name
                        candidate.email = email
                        candidate.phone_number = phone
                        candidate.location = location
                        candidate.position = position
                        candidate.current_status = status
                        candidate.status_due_date = datetime.strptime(
                            due_date, '%Y-%m-%d') if due_date else None
                        candidate.assignee = assignee
                        candidate.fail_stage = fail_stage
                        candidate.failed_reason = failed_reason
                        candidate.source_notes = source_notes
                        candidate.notified = notified
                        candidate.last_updated = datetime.now()

                        db.session.commit()

                        # Create row with updated information
                        row = html.Tr([
                            html.Td(name or "Not specified"),
                            html.Td(email or "Not specified"),
                            html.Td(phone or "Not specified"),
                            html.Td(location or "Not specified"),
                            html.Td(candidate.date.strftime("%Y-%m-%d")),
                            html.Td(status or "CV Review"),
                            html.Td(due_date or "Not set"),
                            html.Td(assignee or "Unassigned"),
                            html.Td(position or "Not specified"),
                            html.Td([
                                dbc.Button("Edit", id={"type": "edit-btn", "index": i},
                                           color="primary", size="sm", className="me-2"),
                                dbc.Button("Delete", id={"type": "delete-btn", "index": i},
                                           color="danger", size="sm"),
                            ])
                        ])
                    else:
                        # Use existing data for other rows
                        row = html.Tr([
                            html.Td(candidate.name or "Not specified"),
                            html.Td(candidate.email or "Not specified"),
                            html.Td(candidate.phone_number or "Not specified"),
                            html.Td(candidate.location or "Not specified"),
                            html.Td(candidate.date.strftime("%Y-%m-%d")),
                            html.Td(candidate.current_status or "CV Review"),
                            html.Td(candidate.status_due_date.strftime(
                                "%Y-%m-%d") if candidate.status_due_date else "Not set"),
                            html.Td(candidate.assignee or "Unassigned"),
                            html.Td(candidate.position or "Not specified"),
                            html.Td([
                                dbc.Button("Edit", id={"type": "edit-btn", "index": i},
                                           color="primary", size="sm", className="me-2"),
                                dbc.Button("Delete", id={"type": "delete-btn", "index": i},
                                           color="danger", size="sm"),
                            ])
                        ])
                    table_rows.append(row)

                # Create the updated table
                table = dbc.Table(
                    [
                        html.Thead(
                            html.Tr([
                                html.Th(col) for col in [
                                    "Name", "Email", "Phone", "Location", "Date Uploaded",
                                    "Current Status", "Status Due Date",
                                    "Assignee", "Position", "Actions"
                                ]
                            ])
                        ),
                        html.Tbody(table_rows)
                    ],
                    bordered=True,
                    hover=True,
                    className="mt-4"
                )

                return False, False, table, dbc.Alert(
                    "Information saved successfully!",
                    color="success",
                    dismissable=True,
                    duration=3000
                )

            except Exception as e:
                print(f"Error saving changes: {str(e)}")  # For debugging
                return False, True, dash.no_update, dbc.Alert(
                    f"Error saving changes: {str(e)}",
                    color="danger",
                    dismissable=True,
                    duration=3000
                )

    except Exception as e:
        return False, False, dash.no_update, dbc.Alert(
            f"An error occurred: {str(e)}",
            color="danger",
            dismissable=True,
            duration=3000
        )

    return upload_is_open, info_is_open, dash.no_update, dash.no_update

# Add a callback to pre-fill the form with extracted information


@app.callback(
    [Output("candidate-name", "value"),
     Output("candidate-email", "value"),
     Output("candidate-phone", "value"),
     Output("candidate-location", "value"),
     Output("candidate-position", "value"),
     Output("current-status", "value"),
     Output("status-due-date", "value"),
     Output("assignee", "value"),
     Output("failed-stage", "value"),
     Output("failed-reason", "value"),
     Output("source-notes", "value"),
     Output("candidate-notified", "value")],
    [Input("info-modal", "is_open")],
    prevent_initial_call=True
)
def prefill_form(is_open):
    if is_open:
        latest_candidate = Candidate.query.order_by(
            Candidate.id.desc()).first()
        if latest_candidate:
            return (
                latest_candidate.name or "",
                latest_candidate.email or "",
                latest_candidate.phone_number or "",
                latest_candidate.location or "",
                latest_candidate.position or "",
                latest_candidate.current_status or "",
                latest_candidate.status_due_date.strftime(
                    "%Y-%m-%d") if latest_candidate.status_due_date else "",
                latest_candidate.assignee or "",
                latest_candidate.fail_stage or "",
                latest_candidate.failed_reason or "",
                latest_candidate.source_notes or "",
                latest_candidate.notified or False
            )
    return "", "", "", "", "", "", "", "", "", "", "", False


@app.callback(
    Output("upload-modal", "is_open", allow_duplicate=True),
    Input("alert-upload", "n_clicks"),
    prevent_initial_call=True
)
def open_upload_from_alert(n_clicks):
    if n_clicks:
        return True
    return False


@app.callback(
    Output("alert-container", "children", allow_duplicate=True),
    Input("table-container", "children"),
    prevent_initial_call=True
)
def update_alert_visibility(table_content):
    if not table_content:
        return dbc.Alert([
            "No candidates in the database ",
            html.A("Upload a CV", href="#",
                   id="alert-upload", className="alert-link"),
            "."
        ], color="info")
    return None


# Add this callback to handle the Back to List button
@app.callback(
    Output("info-modal", "is_open", allow_duplicate=True),
    Input("back-to-list", "n_clicks"),
    prevent_initial_call=True
)
def close_info_modal(n_clicks):
    if n_clicks:
        return False
    return dash.no_update


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run_server(host="0.0.0.0", port=port, debug=False)
