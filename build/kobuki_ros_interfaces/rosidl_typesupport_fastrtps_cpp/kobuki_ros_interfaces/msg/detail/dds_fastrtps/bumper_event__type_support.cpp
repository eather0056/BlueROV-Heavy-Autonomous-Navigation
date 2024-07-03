// generated from rosidl_typesupport_fastrtps_cpp/resource/idl__type_support.cpp.em
// with input from kobuki_ros_interfaces:msg/BumperEvent.idl
// generated code does not contain a copyright notice
#include "kobuki_ros_interfaces/msg/detail/bumper_event__rosidl_typesupport_fastrtps_cpp.hpp"
#include "kobuki_ros_interfaces/msg/detail/bumper_event__struct.hpp"

#include <limits>
#include <stdexcept>
#include <string>
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_fastrtps_cpp/identifier.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_fastrtps_cpp/wstring_conversion.hpp"
#include "fastcdr/Cdr.h"


// forward declaration of message dependencies and their conversion functions

namespace kobuki_ros_interfaces
{

namespace msg
{

namespace typesupport_fastrtps_cpp
{

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_kobuki_ros_interfaces
cdr_serialize(
  const kobuki_ros_interfaces::msg::BumperEvent & ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  // Member: bumper
  cdr << ros_message.bumper;
  // Member: state
  cdr << ros_message.state;
  return true;
}

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_kobuki_ros_interfaces
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  kobuki_ros_interfaces::msg::BumperEvent & ros_message)
{
  // Member: bumper
  cdr >> ros_message.bumper;

  // Member: state
  cdr >> ros_message.state;

  return true;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_kobuki_ros_interfaces
get_serialized_size(
  const kobuki_ros_interfaces::msg::BumperEvent & ros_message,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // Member: bumper
  {
    size_t item_size = sizeof(ros_message.bumper);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: state
  {
    size_t item_size = sizeof(ros_message.state);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_kobuki_ros_interfaces
max_serialized_size_BumperEvent(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  size_t last_member_size = 0;
  (void)last_member_size;
  (void)padding;
  (void)wchar_size;

  full_bounded = true;
  is_plain = true;


  // Member: bumper
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: state
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  size_t ret_val = current_alignment - initial_alignment;
  if (is_plain) {
    // All members are plain, and type is not empty.
    // We still need to check that the in-memory alignment
    // is the same as the CDR mandated alignment.
    using DataType = kobuki_ros_interfaces::msg::BumperEvent;
    is_plain =
      (
      offsetof(DataType, state) +
      last_member_size
      ) == ret_val;
  }

  return ret_val;
}

static bool _BumperEvent__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  auto typed_message =
    static_cast<const kobuki_ros_interfaces::msg::BumperEvent *>(
    untyped_ros_message);
  return cdr_serialize(*typed_message, cdr);
}

static bool _BumperEvent__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  auto typed_message =
    static_cast<kobuki_ros_interfaces::msg::BumperEvent *>(
    untyped_ros_message);
  return cdr_deserialize(cdr, *typed_message);
}

static uint32_t _BumperEvent__get_serialized_size(
  const void * untyped_ros_message)
{
  auto typed_message =
    static_cast<const kobuki_ros_interfaces::msg::BumperEvent *>(
    untyped_ros_message);
  return static_cast<uint32_t>(get_serialized_size(*typed_message, 0));
}

static size_t _BumperEvent__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_BumperEvent(full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}

static message_type_support_callbacks_t _BumperEvent__callbacks = {
  "kobuki_ros_interfaces::msg",
  "BumperEvent",
  _BumperEvent__cdr_serialize,
  _BumperEvent__cdr_deserialize,
  _BumperEvent__get_serialized_size,
  _BumperEvent__max_serialized_size
};

static rosidl_message_type_support_t _BumperEvent__handle = {
  rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
  &_BumperEvent__callbacks,
  get_message_typesupport_handle_function,
};

}  // namespace typesupport_fastrtps_cpp

}  // namespace msg

}  // namespace kobuki_ros_interfaces

namespace rosidl_typesupport_fastrtps_cpp
{

template<>
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_EXPORT_kobuki_ros_interfaces
const rosidl_message_type_support_t *
get_message_type_support_handle<kobuki_ros_interfaces::msg::BumperEvent>()
{
  return &kobuki_ros_interfaces::msg::typesupport_fastrtps_cpp::_BumperEvent__handle;
}

}  // namespace rosidl_typesupport_fastrtps_cpp

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, kobuki_ros_interfaces, msg, BumperEvent)() {
  return &kobuki_ros_interfaces::msg::typesupport_fastrtps_cpp::_BumperEvent__handle;
}

#ifdef __cplusplus
}
#endif