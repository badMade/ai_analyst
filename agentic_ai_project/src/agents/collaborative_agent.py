"""Collaborative Agent implementation.

An agent designed to work cooperatively with other agents,
supporting communication, coordination, and consensus-building.
"""

from dataclasses import dataclass, field
from typing import Any, Callable
from enum import Enum
import asyncio
import logging
import uuid

from .base_agent import BaseAgent, AgentConfig, AgentAction, AgentState

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of inter-agent messages."""
    REQUEST = "request"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    PROPOSAL = "proposal"
    VOTE = "vote"
    CONSENSUS = "consensus"


@dataclass
class Message:
    """Message for inter-agent communication."""
    id: str
    sender: str
    recipient: str | None  # None for broadcasts
    message_type: MessageType
    content: Any
    timestamp: float = 0.0
    in_reply_to: str | None = None


@dataclass
class CollaborativeConfig(AgentConfig):
    """Configuration for collaborative agents."""
    communication_protocol: str = "async"
    consensus_threshold: float = 0.7
    max_agents: int = 10
    message_timeout: float = 30.0
    voting_rounds: int = 3


class CollaborativeAgent(BaseAgent):
    """Agent that collaborates with other agents.

    Implements communication protocols, voting mechanisms,
    and consensus-building for multi-agent coordination.
    """

    # Class-level registry for agent communication
    _agent_registry: dict[str, "CollaborativeAgent"] = {}

    def __init__(
        self,
        name: str | None = None,
        config: CollaborativeConfig | None = None,
        team: list[str] | None = None,
    ):
        """Initialize the collaborative agent.

        Args:
            name: Optional name for the agent.
            config: Configuration settings.
            team: List of teammate agent IDs.
        """
        super().__init__(name, config or CollaborativeConfig())
        self.config: CollaborativeConfig = self.config
        self.team = team or []
        self.inbox: list[Message] = []
        self.outbox: list[Message] = []
        self.pending_proposals: dict[str, dict[str, Any]] = {}
        self.votes_received: dict[str, list[tuple[str, bool]]] = {}

        # Register this agent
        CollaborativeAgent._agent_registry[self.id] = self

    def perceive(self, observation: Any) -> dict[str, Any]:
        """Process observation and incoming messages.

        Args:
            observation: Raw observation from environment.

        Returns:
            Combined perception with messages.
        """
        # Process incoming messages
        messages = self._process_inbox()

        return {
            "observation": observation,
            "messages": messages,
            "pending_proposals": list(self.pending_proposals.keys()),
            "team_status": self._get_team_status(),
        }

    def decide(self, perception: dict[str, Any]) -> AgentAction:
        """Decide on action considering team coordination.

        Args:
            perception: Processed perception with messages.

        Returns:
            Coordinated action.
        """
        messages = perception.get("messages", [])
        pending = perception.get("pending_proposals", [])

        # Handle incoming messages first
        for msg in messages:
            self._handle_message(msg)

        # Check if we need to propose or vote
        if pending:
            return AgentAction(
                action_type="process_proposals",
                parameters={"proposals": pending}
            )

        # Default collaborative action
        return AgentAction(
            action_type="coordinate",
            parameters={"observation": perception.get("observation")}
        )

    def act(self, action: AgentAction) -> Any:
        """Execute collaborative action.

        Args:
            action: Action to execute.

        Returns:
            Action result.
        """
        action_type = action.action_type

        if action_type == "process_proposals":
            return self._process_proposals(action.parameters.get("proposals", []))
        elif action_type == "coordinate":
            return self._coordinate_action(action.parameters)

        return {"success": True, "action": action_type}

    def send_message(
        self,
        recipient: str | None,
        message_type: MessageType,
        content: Any,
        in_reply_to: str | None = None,
    ) -> Message:
        """Send a message to another agent or broadcast.

        Args:
            recipient: Target agent ID, or None for broadcast.
            message_type: Type of message.
            content: Message content.
            in_reply_to: Optional ID of message being replied to.

        Returns:
            The sent message.
        """
        message = Message(
            id=str(uuid.uuid4()),
            sender=self.id,
            recipient=recipient,
            message_type=message_type,
            content=content,
            in_reply_to=in_reply_to,
        )

        self.outbox.append(message)

        # Deliver message
        if recipient:
            self._deliver_to(recipient, message)
        else:
            self._broadcast(message)

        logger.debug(f"{self.name}: Sent {message_type.value} to {recipient or 'all'}")
        return message

    def propose(self, proposal: dict[str, Any]) -> str:
        """Propose an action or decision for team consensus.

        Args:
            proposal: The proposal content.

        Returns:
            Proposal ID.
        """
        proposal_id = str(uuid.uuid4())

        self.pending_proposals[proposal_id] = {
            "content": proposal,
            "votes": [],
            "status": "pending",
        }

        # Broadcast proposal to team
        self.send_message(
            recipient=None,
            message_type=MessageType.PROPOSAL,
            content={"proposal_id": proposal_id, "proposal": proposal},
        )

        logger.info(f"{self.name}: Created proposal {proposal_id[:8]}")
        return proposal_id

    def vote(self, proposal_id: str, approve: bool, reason: str = "") -> None:
        """Vote on a proposal.

        Args:
            proposal_id: ID of the proposal.
            approve: Whether to approve the proposal.
            reason: Optional reason for the vote.
        """
        # Find the proposer
        for agent_id, agent in CollaborativeAgent._agent_registry.items():
            if proposal_id in agent.pending_proposals:
                self.send_message(
                    recipient=agent_id,
                    message_type=MessageType.VOTE,
                    content={
                        "proposal_id": proposal_id,
                        "approve": approve,
                        "reason": reason,
                    },
                )
                break

    def check_consensus(self, proposal_id: str) -> tuple[bool, float]:
        """Check if consensus has been reached on a proposal.

        Args:
            proposal_id: ID of the proposal.

        Returns:
            Tuple of (consensus_reached, approval_ratio).
        """
        if proposal_id not in self.pending_proposals:
            return False, 0.0

        proposal = self.pending_proposals[proposal_id]
        votes = proposal.get("votes", [])

        if not votes:
            return False, 0.0

        approvals = sum(1 for _, approved in votes if approved)
        ratio = approvals / len(votes)

        consensus = ratio >= self.config.consensus_threshold
        return consensus, ratio

    def join_team(self, team_id: str) -> None:
        """Join a team of agents.

        Args:
            team_id: ID of team member to connect with.
        """
        if team_id not in self.team:
            self.team.append(team_id)
            logger.info(f"{self.name}: Joined team with {team_id}")

    def leave_team(self, team_id: str) -> None:
        """Leave a team.

        Args:
            team_id: ID of team to leave.
        """
        if team_id in self.team:
            self.team.remove(team_id)
            logger.info(f"{self.name}: Left team {team_id}")

    def _process_inbox(self) -> list[Message]:
        """Process and clear inbox."""
        messages = self.inbox.copy()
        self.inbox.clear()
        return messages

    def _handle_message(self, message: Message) -> None:
        """Handle an incoming message."""
        if message.message_type == MessageType.PROPOSAL:
            self._handle_proposal(message)
        elif message.message_type == MessageType.VOTE:
            self._handle_vote(message)
        elif message.message_type == MessageType.REQUEST:
            self._handle_request(message)

    def _handle_proposal(self, message: Message) -> None:
        """Handle a proposal message."""
        content = message.content
        proposal_id = content.get("proposal_id")
        proposal = content.get("proposal")

        # Auto-vote for now (can be made more sophisticated)
        approve = self._evaluate_proposal(proposal)
        self.vote(proposal_id, approve, "Automated evaluation")

    def _handle_vote(self, message: Message) -> None:
        """Handle a vote message."""
        content = message.content
        proposal_id = content.get("proposal_id")
        approve = content.get("approve")

        if proposal_id in self.pending_proposals:
            self.pending_proposals[proposal_id]["votes"].append(
                (message.sender, approve)
            )

            # Check for consensus
            consensus, ratio = self.check_consensus(proposal_id)
            if consensus:
                logger.info(f"{self.name}: Consensus reached on {proposal_id[:8]} ({ratio:.0%})")
                self.pending_proposals[proposal_id]["status"] = "approved"

    def _handle_request(self, message: Message) -> None:
        """Handle a request message."""
        # Send response
        self.send_message(
            recipient=message.sender,
            message_type=MessageType.RESPONSE,
            content={"status": "acknowledged"},
            in_reply_to=message.id,
        )

    def _evaluate_proposal(self, proposal: dict[str, Any]) -> bool:
        """Evaluate a proposal for voting."""
        # Placeholder - implement actual evaluation logic
        return True

    def _deliver_to(self, recipient: str, message: Message) -> None:
        """Deliver message to a specific agent."""
        if recipient in CollaborativeAgent._agent_registry:
            CollaborativeAgent._agent_registry[recipient].inbox.append(message)

    def _broadcast(self, message: Message) -> None:
        """Broadcast message to all team members."""
        for team_member in self.team:
            if team_member in CollaborativeAgent._agent_registry:
                CollaborativeAgent._agent_registry[team_member].inbox.append(message)

    def _get_team_status(self) -> dict[str, str]:
        """Get status of team members."""
        status = {}
        for member_id in self.team:
            if member_id in CollaborativeAgent._agent_registry:
                agent = CollaborativeAgent._agent_registry[member_id]
                status[member_id] = agent.state.value
            else:
                status[member_id] = "unknown"
        return status

    def _process_proposals(self, proposals: list[str]) -> dict[str, Any]:
        """Process pending proposals."""
        results = {}
        for proposal_id in proposals:
            consensus, ratio = self.check_consensus(proposal_id)
            results[proposal_id] = {
                "consensus": consensus,
                "approval_ratio": ratio,
            }
        return results

    def _coordinate_action(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """Coordinate an action with the team."""
        return {"coordinated": True, "team_size": len(self.team)}

    @classmethod
    def get_agent(cls, agent_id: str) -> "CollaborativeAgent | None":
        """Get an agent by ID from the registry."""
        return cls._agent_registry.get(agent_id)

    @classmethod
    def clear_registry(cls) -> None:
        """Clear the agent registry."""
        cls._agent_registry.clear()
