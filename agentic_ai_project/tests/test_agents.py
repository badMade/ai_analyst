"""Tests for agent implementations."""

import pytest
from src.agents import (
    BaseAgent,
    AutonomousAgent,
    LearningAgent,
    ReasoningAgent,
    CollaborativeAgent,
)
from src.agents.base_agent import AgentConfig, AgentAction, AgentState


class ConcreteAgent(BaseAgent):
    """Concrete implementation for testing BaseAgent."""

    def perceive(self, observation):
        return {"raw": observation}

    def decide(self, perception):
        return AgentAction(action_type="test", parameters={})

    def act(self, action):
        return {"success": True}


class TestBaseAgent:
    """Tests for BaseAgent."""

    def test_initialization(self):
        """Test agent initialization."""
        agent = ConcreteAgent(name="TestAgent")
        assert agent.name == "TestAgent"
        assert agent.state == AgentState.IDLE
        assert agent.step_count == 0

    def test_step(self):
        """Test agent step execution."""
        agent = ConcreteAgent()
        result = agent.step({"test": "observation"})
        assert agent.step_count == 1
        assert agent.state == AgentState.RUNNING
        assert result is not None

    def test_reset(self):
        """Test agent reset."""
        agent = ConcreteAgent()
        agent.step({"test": "observation"})
        agent.reset()
        assert agent.step_count == 0
        assert agent.state == AgentState.IDLE

    def test_memory(self):
        """Test agent memory operations."""
        agent = ConcreteAgent()
        agent.set_memory("key", "value")
        assert agent.get_memory("key") == "value"
        assert agent.get_memory("nonexistent") is None

    def test_pause_resume(self):
        """Test agent pause and resume."""
        agent = ConcreteAgent()
        agent.step({})
        agent.pause()
        assert agent.state == AgentState.PAUSED
        agent.resume()
        assert agent.state == AgentState.RUNNING

    def test_max_steps(self):
        """Test max steps limit."""
        config = AgentConfig(max_steps=3)
        agent = ConcreteAgent(config=config)

        for i in range(5):
            agent.step({})

        assert agent.step_count == 4
        assert agent.state == AgentState.COMPLETED


class TestAutonomousAgent:
    """Tests for AutonomousAgent."""

    def test_initialization(self):
        """Test autonomous agent initialization."""
        agent = AutonomousAgent(name="AutoAgent", goals=["goal1", "goal2"])
        assert agent.name == "AutoAgent"
        assert len(agent.goals) == 2

    def test_set_goal(self):
        """Test goal setting."""
        agent = AutonomousAgent()
        agent.set_goal("test_goal")
        assert agent.current_goal == "test_goal"
        assert "test_goal" in agent.goals

    def test_goal_progress(self):
        """Test goal progress tracking."""
        agent = AutonomousAgent()
        agent.set_goal("test_goal")
        agent.update_goal_progress(0.5)
        assert agent.goal_progress["test_goal"] == 0.5

    def test_step_execution(self):
        """Test step execution."""
        agent = AutonomousAgent()
        agent.set_goal("test_goal")
        result = agent.step({"state": "test"})
        assert agent.step_count == 1


class TestLearningAgent:
    """Tests for LearningAgent."""

    def test_initialization(self):
        """Test learning agent initialization."""
        agent = LearningAgent(name="LearnAgent")
        assert agent.name == "LearnAgent"
        assert agent.training_step == 0

    def test_experience_buffer(self):
        """Test experience replay buffer."""
        agent = LearningAgent()
        agent.step({})

        # Add experience
        agent.learn(
            state={"s": 1},
            action=AgentAction(action_type="test", parameters={}),
            reward=1.0,
            next_state={"s": 2},
            done=False,
        )

        assert len(agent.replay_buffer) == 1

    def test_training_stats(self):
        """Test training statistics."""
        agent = LearningAgent()
        stats = agent.get_training_stats()
        assert "training_steps" in stats
        assert "buffer_size" in stats


class TestReasoningAgent:
    """Tests for ReasoningAgent."""

    def test_initialization(self):
        """Test reasoning agent initialization."""
        agent = ReasoningAgent(name="ReasonAgent")
        assert agent.name == "ReasonAgent"

    def test_knowledge_base(self):
        """Test knowledge base operations."""
        agent = ReasoningAgent()
        agent.add_knowledge("fact1", "The sky is blue")
        assert "fact1" in agent.knowledge_base

    def test_reasoning(self):
        """Test reasoning execution."""
        agent = ReasoningAgent()
        agent.add_knowledge("premise", "All humans are mortal")

        trace = agent.reason("Is Socrates mortal?")
        assert trace.query == "Is Socrates mortal?"
        assert len(trace.thoughts) > 0
        assert trace.confidence > 0

    def test_step_with_reasoning(self):
        """Test step with reasoning."""
        agent = ReasoningAgent()
        result = agent.step("What is 2+2?")
        assert agent.step_count == 1


class TestCollaborativeAgent:
    """Tests for CollaborativeAgent."""

    def test_initialization(self):
        """Test collaborative agent initialization."""
        agent = CollaborativeAgent(name="CollabAgent")
        assert agent.name == "CollabAgent"
        assert len(agent.team) == 0

    def test_team_management(self):
        """Test team joining and leaving."""
        agent = CollaborativeAgent()
        agent.join_team("team1")
        assert "team1" in agent.team
        agent.leave_team("team1")
        assert "team1" not in agent.team

    def test_messaging(self):
        """Test inter-agent messaging."""
        from src.agents.collaborative_agent import MessageType

        agent1 = CollaborativeAgent(name="Agent1")
        agent2 = CollaborativeAgent(name="Agent2")
        agent1.join_team(agent2.id)

        message = agent1.send_message(
            recipient=agent2.id,
            message_type=MessageType.REQUEST,
            content={"action": "help"},
        )

        assert message.sender == agent1.id
        assert len(agent2.inbox) == 1

    def test_proposal_and_voting(self):
        """Test proposal and voting mechanism."""
        agent1 = CollaborativeAgent(name="Agent1")
        agent2 = CollaborativeAgent(name="Agent2")

        agent1.join_team(agent2.id)
        agent2.join_team(agent1.id)

        proposal_id = agent1.propose({"action": "collaborate"})
        assert proposal_id in agent1.pending_proposals

    def test_cleanup(self):
        """Clean up agent registry after tests."""
        CollaborativeAgent.clear_registry()
        assert len(CollaborativeAgent._agent_registry) == 0
