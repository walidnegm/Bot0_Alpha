#!/bin/bash

# Colors for terminal output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting all services...${NC}\n"

# Debug current directory
echo "Current directory: $(pwd)"

# Function to check if a port is in use
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null ; then
        echo -e "${RED}Port $1 is already in use. Please free up the port and try again.${NC}"
        exit 1
    fi
}

# Check all ports before starting
check_port 4000  # React
check_port 8000  # LLM Service
check_port 8001  # Interview Agent
check_port 8002  # Resume Service 

# Start React frontend on port 4000
echo -e "${GREEN}Starting React frontend on port 4000...${NC}"
cd /root/my-react-app && PORT=4000 npm start &
REACT_PID=$!

# Start FastAPI LLM service on port 8000
echo -e "${GREEN}Starting FastAPI LLM service on port 8000...${NC}"
cd /root/backend && uvicorn llm_service:app --reload --port 8000 &
LLM_PID=$!

# Start FastAPI Interview Agent on port 8001
echo -e "${GREEN}Starting FastAPI Interview Agent on port 8001...${NC}"
cd /root/backend && uvicorn interviewagent:app --reload --port 8001 &
INTERVIEW_PID=$!

# Start FastAPI Interview Agent on port 8002
echo -e "${GREEN}Starting FastAPI resume service Agent on port 8002...${NC}"
cd /root/backend && uvicorn resume_service:app --reload --port 8002 &
#INTERVIEW_PID=$!

# Store PIDs in a file for cleanup
cd /root
echo $REACT_PID > .running_services
echo $LLM_PID >> .running_services
echo $INTERVIEW_PID >> .running_services

echo -e "\n${BLUE}All services started:${NC}"
echo -e "  • React frontend (http://localhost:4000)"
echo -e "  • LLM service (http://localhost:8000)"
echo -e "  • Interview Agent (http://localhost:8001)"
echo -e "  • Interview Agent (http://localhost:8002)"
echo -e "\n${BLUE}Press Ctrl+C to stop all services${NC}"

# Function to clean up processes
cleanup() {
    echo -e "\n${BLUE}Stopping all services...${NC}"
    if [ -f /root/.running_services ]; then
        while read pid; do
            if ps -p $pid > /dev/null; then
                kill $pid
                echo "Stopped process $pid"
            fi
        done < /root/.running_services
        rm /root/.running_services
    fi
    exit 0
}

# Set up trap to catch Ctrl+C and clean up
trap cleanup SIGINT SIGTERM

# Wait for all background processes
wait
